from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from .ai import AIServiceError, OpenAIVisionService, PreparedPhoto
from .config import Settings
from .costs import BudgetExceeded, CostLedger
from .ign import IGNClient
from .imagery import IGNImageryClient, ImageryError
from .models import Candidate, ListingProfile, RerankResult, SearchResult
from .scoring import (
    apply_luna_scores,
    apply_terra_scores,
    build_and_score_candidates,
    is_ambiguous,
)

StatusCallback = Callable[[str, float], None]


def run_search(
    *,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    profile: ListingProfile,
    photos: list[PreparedPhoto],
    settings: Settings,
    ign: IGNClient,
    imagery: IGNImageryClient,
    ledger: CostLedger,
    ai: OpenAIVisionService | None,
    use_terra_when_ambiguous: bool,
    status: StatusCallback | None = None,
) -> SearchResult:
    warnings: list[str] = []
    update(status, "Gebouwcontouren binnen de echte cirkel ophalen", 0.08)
    buildings = ign.fetch_buildings(
        center_lat,
        center_lon,
        radius_m,
        progress=lambda message: update(status, message, 0.16),
    )
    update(status, "Lokale geometrische kenmerken en omgeving berekenen", 0.25)
    candidates = build_and_score_candidates(
        buildings,
        profile,
        center_lat,
        center_lon,
        radius_m,
    )
    if not candidates:
        return SearchResult(
            candidates=[],
            total_buildings=len(buildings),
            shortlist_size=0,
            contact_sheet_path=None,
            warnings=["Geen geschikte gebouwcontouren binnen de gekozen cirkel gevonden."],
        )

    shortlist = candidates[: settings.local_shortlist_size]
    update(status, "Hoge-resolutie luchtfotocrops voorbereiden", 0.36)
    warnings.extend(
        imagery.prefetch_for_candidates(
            shortlist,
            settings.imagery_zoom,
            settings.imagery_crop_size,
        )[:5]
    )

    crops: dict[str, Image.Image] = {}
    usable_shortlist: list[Candidate] = []
    for index, candidate in enumerate(shortlist, start=1):
        try:
            crop, path = imagery.candidate_crop(candidate)
            candidate.crop_path = str(path)
            crops[candidate.candidate_id] = crop
            usable_shortlist.append(candidate)
        except ImageryError as exc:
            warnings.append(str(exc))
        update(
            status,
            f"Luchtfotocrop {index} van {len(shortlist)} opgebouwd",
            0.36 + 0.22 * index / max(1, len(shortlist)),
        )

    contact_sheet_path: str | None = None
    contact_sheet: Image.Image | None = None
    if usable_shortlist:
        contact_sheet, sheet_path = imagery.contact_sheet(usable_shortlist, crops)
        contact_sheet_path = str(sheet_path)

    if ai and photos and contact_sheet and usable_shortlist:
        update(status, "Luna vergelijkt de shortlist in één budgetbegrensde call", 0.64)
        try:
            luna_result = ai.rerank_candidates(
                candidates=usable_shortlist,
                photos=photos,
                contact_sheet=contact_sheet,
                profile=profile,
                ledger=ledger,
                model=settings.luna_model,
                label="Luna-kandidatenvergelijking",
                reasoning_effort="none",
            )
            apply_assessments(usable_shortlist, luna_result)
            apply_luna_scores(
                usable_shortlist,
                {
                    assessment.candidate_id: assessment.visual_score
                    for assessment in luna_result.assessments
                },
            )
        except (BudgetExceeded, AIServiceError) as exc:
            warnings.append(f"Luna-reranking overgeslagen: {exc}")

    candidates.sort(key=lambda item: item.final_score, reverse=True)
    terra_pool = [candidate for candidate in candidates if candidate.crop_path][
        : settings.terra_shortlist_size
    ]
    if (
        ai
        and photos
        and use_terra_when_ambiguous
        and len(terra_pool) >= 2
        and is_ambiguous(terra_pool)
    ):
        update(
            status, "Topkandidaten liggen dicht bij elkaar; Terra voert één eindcontrole uit", 0.82
        )
        try:
            terra_crops = {
                candidate.candidate_id: crops[candidate.candidate_id]
                for candidate in terra_pool
                if candidate.candidate_id in crops
            }
            terra_pool = [
                candidate for candidate in terra_pool if candidate.candidate_id in terra_crops
            ]
            terra_sheet, _ = imagery.contact_sheet(terra_pool, terra_crops, columns=2)
            terra_result = ai.rerank_candidates(
                candidates=terra_pool,
                photos=photos,
                contact_sheet=terra_sheet,
                profile=profile,
                ledger=ledger,
                model=settings.terra_model,
                label="Terra-eindcontrole",
                reasoning_effort="low",
            )
            apply_assessments(terra_pool, terra_result)
            apply_terra_scores(
                terra_pool,
                {
                    assessment.candidate_id: assessment.visual_score
                    for assessment in terra_result.assessments
                },
            )
        except (BudgetExceeded, AIServiceError, ImageryError) as exc:
            warnings.append(f"Terra-eindcontrole overgeslagen: {exc}")

    candidates.sort(key=lambda item: item.final_score, reverse=True)
    update(status, "Zoekopdracht afgerond", 1.0)
    return SearchResult(
        candidates=candidates[: settings.displayed_candidates],
        total_buildings=len(buildings),
        shortlist_size=len(usable_shortlist),
        contact_sheet_path=contact_sheet_path,
        warnings=deduplicate(warnings)[:10],
    )


def apply_assessments(candidates: list[Candidate], result: RerankResult) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for assessment in result.assessments:
        candidate = by_id.get(assessment.candidate_id)
        if not candidate:
            continue
        candidate.reasons = deduplicate(candidate.reasons + assessment.supporting_clues)[:7]
        candidate.conflicts = deduplicate(candidate.conflicts + assessment.conflicts)[:7]
        candidate.uncertainties = deduplicate(candidate.uncertainties + assessment.uncertainties)[
            :7
        ]


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def update(status: StatusCallback | None, message: str, progress: float) -> None:
    if status:
        status(message, min(1.0, max(0.0, progress)))
