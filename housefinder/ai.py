from __future__ import annotations

import base64
import io
from collections.abc import Iterable
from dataclasses import dataclass

from openai import OpenAI
from PIL import Image, ImageOps
from pydantic import BaseModel

from .config import Settings
from .costs import CostLedger
from .models import Candidate, ListingProfile, RerankResult


class AIServiceError(RuntimeError):
    pass


@dataclass
class PreparedPhoto:
    index: int
    image: Image.Image
    data_url: str


def prepare_photos(
    image_bytes: Iterable[bytes], max_photos: int, max_side: int
) -> list[PreparedPhoto]:
    prepared: list[PreparedPhoto] = []
    for index, raw in enumerate(list(image_bytes)[:max_photos], start=1):
        try:
            image = Image.open(io.BytesIO(raw))
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            prepared.append(
                PreparedPhoto(index=index, image=image, data_url=image_to_data_url(image))
            )
        except OSError:
            continue
    return prepared


def image_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class OpenAIVisionService:
    def __init__(self, api_key: str, settings: Settings):
        self.client = OpenAI(api_key=api_key)
        self.settings = settings

    def analyze_listing_photos(
        self, photos: list[PreparedPhoto], ledger: CostLedger
    ) -> ListingProfile:
        if not photos:
            raise AIServiceError("Er zijn geen leesbare woningfoto's aangeleverd.")
        prompt = (
            "Analyseer de aangeleverde makelaarsfoto's uitsluitend voor kenmerken die later "
            "op een noordgerichte luchtfoto bruikbaar kunnen zijn. Een niet-zichtbaar object is "
            "geen bewijs dat het niet bestaat. Gebruik daarom 'unknown' en lage confidence als "
            "een kenmerk niet betrouwbaar zichtbaar is. Schat geen exacte coördinaten. "
            "outbuildings_visible is alleen het aantal daadwerkelijk zichtbare losse gebouwen. "
            "absence_of_outbuildings_conclusive mag alleen true zijn als de foto's aantoonbaar de "
            "volledige kavel rondom tonen. Kies maximaal vier foto-indices met de meeste ruimtelijke "
            "informatie. Antwoord in het gevraagde schema."
        )
        max_output_tokens = 850
        images = [photo.image for photo in photos]
        estimated = ledger.ensure_affordable(
            "Fotoanalyse",
            self.settings.luna_model,
            images,
            prompt,
            max_output_tokens,
        )
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for photo in photos:
            content.extend(
                [
                    {"type": "input_text", "text": f"Woningfoto {photo.index}"},
                    {
                        "type": "input_image",
                        "image_url": photo.data_url,
                        "detail": "original",
                    },
                ]
            )
        response = self._structured_response(
            model=self.settings.luna_model,
            schema_model=ListingProfile,
            schema_name="listing_profile",
            content=content,
            max_output_tokens=max_output_tokens,
            reasoning_effort="none",
        )
        self._record_usage(
            ledger,
            "Fotoanalyse",
            self.settings.luna_model,
            response,
            estimated,
            images,
            prompt,
            max_output_tokens,
        )
        try:
            return ListingProfile.model_validate_json(response.output_text)
        except Exception as exc:
            raise AIServiceError(f"Ongeldig profielantwoord van Luna: {exc}") from exc

    def rerank_candidates(
        self,
        *,
        candidates: list[Candidate],
        photos: list[PreparedPhoto],
        contact_sheet: Image.Image,
        profile: ListingProfile,
        ledger: CostLedger,
        model: str,
        label: str,
        reasoning_effort: str,
    ) -> RerankResult:
        candidate_lines = [
            (
                f"{candidate.candidate_id}: area={candidate.area_m2:.0f}m2, "
                f"shape={candidate.shape_class}, buildings_50m={candidate.buildings_within_50m}, "
                f"buildings_100m={candidate.buildings_within_100m}, local_score={candidate.local_score:.1f}"
            )
            for candidate in candidates
        ]
        prompt = (
            "Vergelijk de originele woningfoto's met de gelabelde noordgerichte luchtfoto-crops. "
            "De rode contour is het kandidaatgebouw en de witte kruising markeert het midden. "
            "Beoordeel uitsluitend structurele, vanuit de lucht zichtbare aanwijzingen: gebouwvorm, "
            "relatie tot bijgebouwen, weg/oprit, zwembad, boomlijnen, veldranden en bebouwingsdichtheid. "
            "Vergelijk kandidaten onderling. Geef iedere vermelde candidate_id precies eenmaal terug. "
            "Een niet-zichtbaar element op een makelaarsfoto is geen harde tegenstrijdigheid. "
            "Vermijd schijnprecisie en benoem onzekerheid.\n\n"
            f"Bevestigd profiel:\n{profile.model_dump_json()}\n\n"
            "Kandidaatmetadata:\n" + "\n".join(candidate_lines)
        )
        max_output_tokens = 1_200
        selected_photos = select_diagnostic_photos(photos, profile.diagnostic_photo_indices)
        input_images = [photo.image for photo in selected_photos] + [contact_sheet]
        estimated = ledger.ensure_affordable(label, model, input_images, prompt, max_output_tokens)
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for photo in selected_photos:
            content.extend(
                [
                    {"type": "input_text", "text": f"Woningfoto {photo.index}"},
                    {
                        "type": "input_image",
                        "image_url": photo.data_url,
                        "detail": "original",
                    },
                ]
            )
        content.extend(
            [
                {"type": "input_text", "text": "Kandidatenoverzicht"},
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(contact_sheet),
                    "detail": "original",
                },
            ]
        )
        response = self._structured_response(
            model=model,
            schema_model=RerankResult,
            schema_name="candidate_ranking",
            content=content,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._record_usage(
            ledger,
            label,
            model,
            response,
            estimated,
            input_images,
            prompt,
            max_output_tokens,
        )
        try:
            parsed = RerankResult.model_validate_json(response.output_text)
        except Exception as exc:
            raise AIServiceError(f"Ongeldige kandidaatranking van {model}: {exc}") from exc

        expected_ids = {candidate.candidate_id for candidate in candidates}
        parsed.assessments = [
            assessment
            for assessment in parsed.assessments
            if assessment.candidate_id in expected_ids
        ]
        return parsed

    def _structured_response(
        self,
        *,
        model: str,
        schema_model: type[BaseModel],
        schema_name: str,
        content: list[dict],
        max_output_tokens: int,
        reasoning_effort: str,
    ):
        try:
            return self.client.responses.create(
                model=model,
                instructions=(
                    "Je bent een voorzichtige geospatiale beeldanalist. Scheid observatie van "
                    "aanname en volg het uitvoerschema exact."
                ),
                input=[{"role": "user", "content": content}],
                reasoning={"effort": reasoning_effort},
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_model.model_json_schema(),
                    },
                },
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except Exception as exc:
            raise AIServiceError(f"OpenAI-aanroep met {model} mislukt: {exc}") from exc

    def _record_usage(
        self,
        ledger: CostLedger,
        label: str,
        model: str,
        response,
        estimated_cost: float,
        images: list[Image.Image],
        prompt: str,
        max_output_tokens: int,
    ) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            ledger.record(label, model, input_tokens, output_tokens)
            return
        estimated_input, estimated_output, _ = ledger.estimate_call(
            model, images, prompt, max_output_tokens
        )
        ledger.record(
            label,
            model,
            estimated_input,
            estimated_output,
            estimated=True,
        )


def select_diagnostic_photos(
    photos: list[PreparedPhoto], requested_indices: list[int]
) -> list[PreparedPhoto]:
    requested = set(requested_indices)
    selected = [photo for photo in photos if photo.index in requested]
    if not selected:
        selected = photos[:4]
    return selected[:4]
