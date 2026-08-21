from __future__ import annotations

import hashlib
import html
import os
from dataclasses import replace
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from housefinder.ai import AIServiceError, OpenAIVisionService, prepare_photos
from housefinder.config import DEFAULT_SETTINGS
from housefinder.costs import BudgetExceeded, CostLedger
from housefinder.geo import extract_coords_from_url
from housefinder.ign import IGNClient, IGNError
from housefinder.imagery import IGNImageryClient
from housefinder.models import ListingProfile, SearchResult
from housefinder.pipeline import run_search

PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS = replace(
    DEFAULT_SETTINGS,
    cache_dir=Path(os.getenv("HOUSEFINDER_CACHE_DIR", PROJECT_DIR / ".cache" / "housefinder")),
)

SIZE_OPTIONS = ["unknown", "small", "medium", "large"]
SHAPE_OPTIONS = ["unknown", "compact", "elongated", "complex", "l_shape", "u_shape", "courtyard"]
SETTING_OPTIONS = ["unknown", "isolated", "small_cluster", "village"]
VEGETATION_OPTIONS = ["unknown", "open", "mixed", "wooded"]
ROAD_OPTIONS = ["unknown", "roadside", "short_drive", "long_drive"]
TERNARY_OPTIONS = ["unknown", "yes", "no"]

LABELS = {
    "unknown": "Onbekend",
    "small": "Klein",
    "medium": "Middelgroot",
    "large": "Groot",
    "compact": "Compact/rechthoekig",
    "elongated": "Langwerpig",
    "complex": "Complex samengesteld",
    "l_shape": "L-vorm",
    "u_shape": "U-vorm",
    "courtyard": "Binnenplaats",
    "isolated": "Vrijstaand/geïsoleerd",
    "small_cluster": "Klein bebouwingscluster",
    "village": "Dorp/dichte bebouwing",
    "open": "Open terrein",
    "mixed": "Gemengde begroeiing",
    "wooded": "Bosrijk",
    "roadside": "Direct aan de weg",
    "short_drive": "Korte oprit",
    "long_drive": "Lange oprit",
    "yes": "Ja",
    "no": "Nee",
}


st.set_page_config(page_title="HouseFinder v2", page_icon="🏡", layout="wide")


@st.cache_resource
def get_data_clients(cache_dir: str) -> tuple[IGNClient, IGNImageryClient]:
    settings = replace(SETTINGS, cache_dir=Path(cache_dir))
    return IGNClient(settings), IGNImageryClient(settings)


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    return str(key or os.getenv("OPENAI_API_KEY") or "").strip() or None


def file_signature(files) -> str:
    digest = hashlib.sha256()
    for uploaded in files or []:
        digest.update(uploaded.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def selection_signature(lat: float, lon: float, radius: float, photo_signature: str) -> str:
    return f"{lat:.7f}|{lon:.7f}|{radius:.1f}|{photo_signature}"


def initialize_state() -> None:
    defaults = {
        "search_lat": 44.891237,
        "search_lon": 1.832689,
        "profile": ListingProfile.unknown().model_dump(),
        "profile_revision": 0,
        "photo_signature": "",
        "ledger": None,
        "search_result": None,
        "result_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # Map clicks happen after the latitude/longitude widgets are rendered.
    # Apply them at the start of the next run, before those widgets exist.
    pending_center = st.session_state.pop("pending_search_center", None)
    if pending_center:
        st.session_state.search_lat = float(pending_center["lat"])
        st.session_state.search_lon = float(pending_center["lon"])
        st.session_state.search_result = None
        st.session_state.result_signature = None


def translated_selectbox(label: str, options: list[str], value: str, key: str) -> str:
    return st.selectbox(
        label,
        options,
        index=options.index(value) if value in options else 0,
        format_func=lambda item: LABELS.get(item, item),
        key=key,
    )


def confirmed_confidence(old_value: str, new_value: str, old_confidence: float) -> float:
    if new_value == "unknown":
        return 0.0
    if new_value != old_value:
        return 0.95
    return old_confidence


def street_map(lat: float, lon: float, zoom: int = 14) -> folium.Map:
    """Regular road map for selecting the search-circle centre."""
    return folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )


def satellite_map(lat: float, lon: float, zoom: int = 15) -> folium.Map:
    map_object = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles=(
            "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg"
            "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
        ),
        attr="IGN BD ORTHO",
        name="IGN-luchtfoto",
        overlay=False,
        control=True,
    ).add_to(map_object)
    return map_object


initialize_state()
ign, imagery = get_data_clients(str(SETTINGS.cache_dir))
api_key = get_api_key()

st.title("🏡 HouseFinder v2")
st.caption(
    "Alle gebouwen binnen de echte cirkel worden eerst met open Franse geo-data gevonden. "
    "AI rangschikt alleen een kleine shortlist, met een harde kostengrens."
)

with st.sidebar:
    st.header("1. Woningfoto's")
    uploaded_files = st.file_uploader(
        "Upload maximaal zes bruikbare buitenfoto's",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > SETTINGS.max_photos:
        st.info(f"Alleen de eerste {SETTINGS.max_photos} foto's worden voor AI gebruikt.")

    current_photo_signature = file_signature(uploaded_files)
    if current_photo_signature != st.session_state.photo_signature:
        st.session_state.photo_signature = current_photo_signature
        st.session_state.profile = ListingProfile.unknown().model_dump()
        st.session_state.profile_revision += 1
        st.session_state.ledger = None
        st.session_state.search_result = None
        st.session_state.result_signature = None

    st.header("2. Zoekcirkel")
    place_query = st.text_input("Plaatsnaam", placeholder="Bijvoorbeeld Aynac")
    if st.button("Zoek plaats", use_container_width=True):
        result = ign.geocode(place_query)
        if result:
            lat, lon, label = result
            st.session_state.search_lat = lat
            st.session_state.search_lon = lon
            st.success(label)
            st.rerun()
        else:
            st.error("Plaats niet gevonden via de openbare Franse geocoder.")

    listing_url = st.text_input("Of plak een Bien'ici-/Google Maps-link")
    if st.button("Neem middelpunt over", use_container_width=True):
        coords = extract_coords_from_url(listing_url)
        if coords:
            st.session_state.search_lat, st.session_state.search_lon = coords
            st.rerun()
        else:
            st.error("In deze link zijn geen herkenbare coördinaten gevonden.")

    st.number_input(
        "Latitude", min_value=41.0, max_value=52.0, step=0.0001, format="%.6f", key="search_lat"
    )
    st.number_input(
        "Longitude", min_value=-6.0, max_value=11.0, step=0.0001, format="%.6f", key="search_lon"
    )
    radius_m = st.slider("Straal van de makelaarscirkel", 100, 3_000, 600, 50, format="%d m")

    st.header("3. Kosten")
    budget_usd = st.slider(
        "Maximale AI-kosten per woning",
        min_value=0.01,
        max_value=0.08,
        value=SETTINGS.default_budget_usd,
        step=0.01,
        format="$%.2f",
    )
    use_terra = st.checkbox(
        "Terra-eindcontrole bij twijfel",
        value=True,
        help="Wordt alleen uitgevoerd als de beste kandidaten dicht bij elkaar liggen én binnen het budget past.",
    )
    if api_key:
        st.success("OpenAI-sleutel gevonden")
    else:
        st.warning(
            "Geen OpenAI-sleutel: de app blijft werken met handmatige aanwijzingen en lokale scoring."
        )


if uploaded_files:
    preview_columns = st.columns(min(3, len(uploaded_files)))
    for index, uploaded in enumerate(uploaded_files[: SETTINGS.max_photos]):
        with preview_columns[index % len(preview_columns)]:
            st.image(uploaded, caption=f"Foto {index + 1}", width="stretch")

prepared_photos = prepare_photos(
    [uploaded.getvalue() for uploaded in uploaded_files or []],
    SETTINGS.max_photos,
    SETTINGS.max_photo_side,
)

st.subheader("Stap 1 · Aanwijzingen uit de foto's")
analysis_col, cost_col = st.columns([2, 1])
with analysis_col:
    analyze_disabled = not prepared_photos or not api_key
    if st.button(
        "Foto's goedkoop analyseren met Luna",
        type="primary",
        disabled=analyze_disabled,
        use_container_width=True,
    ):
        ledger = CostLedger.from_dict(st.session_state.ledger, budget_usd)
        ai_service = OpenAIVisionService(api_key, SETTINGS)
        try:
            with st.spinner("Alleen vanuit de lucht bruikbare aanwijzingen bepalen..."):
                profile = ai_service.analyze_listing_photos(prepared_photos, ledger)
            st.session_state.profile = profile.model_dump()
            st.session_state.profile_revision += 1
            st.session_state.ledger = ledger.to_dict()
            st.session_state.search_result = None
            st.success("Fotoanalyse voltooid. Controleer de aanwijzingen hieronder.")
            st.rerun()
        except (AIServiceError, BudgetExceeded) as exc:
            st.error(str(exc))

with cost_col:
    current_ledger = CostLedger.from_dict(st.session_state.ledger, budget_usd)
    st.metric("AI-kosten tot nu toe", f"${current_ledger.total_usd:.4f}")
    st.caption(f"Resterend hard budget: ${current_ledger.remaining_usd:.4f}")

profile = ListingProfile.model_validate(st.session_state.profile)
revision = st.session_state.profile_revision
with st.expander("Controleer en corrigeer de aanwijzingen", expanded=True):
    st.caption(
        "Een handmatige wijziging krijgt hoge betrouwbaarheid. Zet ‘geen bijgebouwen’ alleen als harde "
        "aanwijzing wanneer de foto's aantoonbaar de volledige kavel tonen."
    )
    with st.form(f"profile_form_{revision}"):
        left, right = st.columns(2)
        with left:
            size_value = translated_selectbox(
                "Relatieve grootte", SIZE_OPTIONS, profile.size_category, f"size_{revision}"
            )
            shape_value = translated_selectbox(
                "Vermoedelijke voetafdruk",
                SHAPE_OPTIONS,
                profile.footprint_shape,
                f"shape_{revision}",
            )
            outbuildings_value = st.number_input(
                "Zichtbare losse bijgebouwen",
                0,
                9,
                profile.outbuildings_visible,
                key=f"outbuildings_{revision}",
            )
            absence_conclusive = st.checkbox(
                "Afwezigheid van bijgebouwen is echt overtuigend",
                value=profile.absence_of_outbuildings_conclusive,
                key=f"absence_{revision}",
            )
        with right:
            setting_value = translated_selectbox(
                "Bebouwingscontext", SETTING_OPTIONS, profile.setting, f"setting_{revision}"
            )
            vegetation_value = translated_selectbox(
                "Vegetatie", VEGETATION_OPTIONS, profile.vegetation, f"vegetation_{revision}"
            )
            road_value = translated_selectbox(
                "Weg/oprit", ROAD_OPTIONS, profile.road_context, f"road_{revision}"
            )
            pool_value = translated_selectbox(
                "Zwembad zichtbaar", TERNARY_OPTIONS, profile.pool_visible, f"pool_{revision}"
            )
        st.write(profile.summary)
        save_profile = st.form_submit_button("Aanwijzingen opslaan", use_container_width=True)

    if save_profile:
        updated = profile.model_copy(
            update={
                "size_category": size_value,
                "size_confidence": confirmed_confidence(
                    profile.size_category, size_value, profile.size_confidence
                ),
                "footprint_shape": shape_value,
                "shape_confidence": confirmed_confidence(
                    profile.footprint_shape, shape_value, profile.shape_confidence
                ),
                "outbuildings_visible": int(outbuildings_value),
                "outbuildings_confidence": (
                    0.95
                    if int(outbuildings_value) != profile.outbuildings_visible
                    else profile.outbuildings_confidence
                ),
                "absence_of_outbuildings_conclusive": absence_conclusive,
                "setting": setting_value,
                "setting_confidence": confirmed_confidence(
                    profile.setting, setting_value, profile.setting_confidence
                ),
                "vegetation": vegetation_value,
                "vegetation_confidence": confirmed_confidence(
                    profile.vegetation, vegetation_value, profile.vegetation_confidence
                ),
                "road_context": road_value,
                "road_confidence": confirmed_confidence(
                    profile.road_context, road_value, profile.road_confidence
                ),
                "pool_visible": pool_value,
                "pool_confidence": confirmed_confidence(
                    profile.pool_visible, pool_value, profile.pool_confidence
                ),
                "summary": "Door gebruiker gecontroleerd profiel.",
            }
        )
        st.session_state.profile = updated.model_dump()
        st.session_state.profile_revision += 1
        st.session_state.search_result = None
        st.success("Aanwijzingen opgeslagen.")
        st.rerun()

st.subheader("Stap 2 · Controleer de exacte cirkel")
st.caption("Klik op de kaart om de rode speld en zoekcirkel te verplaatsen.")
selection_map = street_map(st.session_state.search_lat, st.session_state.search_lon, 14)
folium.Circle(
    location=[st.session_state.search_lat, st.session_state.search_lon],
    radius=radius_m,
    color="#ef4444",
    weight=3,
    fill=True,
    fill_opacity=0.12,
    tooltip=f"Werkelijk zoekgebied: straal {radius_m} m",
).add_to(selection_map)
folium.Marker(
    [st.session_state.search_lat, st.session_state.search_lon],
    tooltip="Middelpunt",
    icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
).add_to(selection_map)
selection_data = st_folium(selection_map, height=420, use_container_width=True, key="selection_map")
if selection_data and selection_data.get("last_clicked"):
    click = selection_data["last_clicked"]
    if (
        abs(click["lat"] - st.session_state.search_lat) > 1e-7
        or abs(click["lng"] - st.session_state.search_lon) > 1e-7
    ):
        st.session_state.pending_search_center = {
            "lat": float(click["lat"]),
            "lon": float(click["lng"]),
        }
        st.rerun()

run_disabled = not (
    41.0 <= st.session_state.search_lat <= 52.0 and -6.0 <= st.session_state.search_lon <= 11.0
)
if st.button(
    "Zoek en rangschik alle gebouwen binnen de cirkel",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
):
    ledger = CostLedger.from_dict(st.session_state.ledger, budget_usd)
    ai_service = OpenAIVisionService(api_key, SETTINGS) if api_key else None
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_status(message: str, progress: float) -> None:
        status_text.write(message)
        progress_bar.progress(progress)

    try:
        result = run_search(
            center_lat=st.session_state.search_lat,
            center_lon=st.session_state.search_lon,
            radius_m=radius_m,
            profile=ListingProfile.model_validate(st.session_state.profile),
            photos=prepared_photos,
            settings=SETTINGS,
            ign=ign,
            imagery=imagery,
            ledger=ledger,
            ai=ai_service,
            use_terra_when_ambiguous=use_terra,
            status=update_status,
        )
        st.session_state.search_result = result
        st.session_state.ledger = ledger.to_dict()
        st.session_state.result_signature = selection_signature(
            st.session_state.search_lat,
            st.session_state.search_lon,
            radius_m,
            st.session_state.photo_signature,
        )
        st.rerun()
    except IGNError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

result: SearchResult | None = st.session_state.search_result
if result:
    st.divider()
    st.subheader("Resultaten")
    active_signature = selection_signature(
        st.session_state.search_lat,
        st.session_state.search_lon,
        radius_m,
        st.session_state.photo_signature,
    )
    if active_signature != st.session_state.result_signature:
        st.warning(
            "De foto's of cirkel zijn gewijzigd. Start de zoekopdracht opnieuw voor actuele resultaten."
        )

    ledger = CostLedger.from_dict(st.session_state.ledger, budget_usd)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Gebouwen in cirkel", result.total_buildings)
    metric_columns[1].metric("AI-shortlist", result.shortlist_size)
    metric_columns[2].metric("Getoonde kandidaten", len(result.candidates))
    metric_columns[3].metric("Werkelijke AI-kosten", f"${ledger.total_usd:.4f}")

    for warning in result.warnings:
        st.warning(warning)

    results_map = satellite_map(st.session_state.search_lat, st.session_state.search_lon, 15)
    folium.Circle(
        [st.session_state.search_lat, st.session_state.search_lon],
        radius=radius_m,
        color="#2563eb",
        weight=3,
        fill=False,
    ).add_to(results_map)
    for rank, candidate in enumerate(result.candidates, start=1):
        maps_link = (
            f"https://www.google.com/maps/search/?api=1&query={candidate.lat},{candidate.lon}"
        )
        popup = (
            f"<b>#{rank} · {html.escape(candidate.candidate_id)}</b><br>"
            f"Score: {candidate.final_score:.1f}/100<br>"
            f"Voetafdruk: {candidate.area_m2:.0f} m²<br>"
            f"<a href='{maps_link}' target='_blank'>Open in Google Maps</a>"
        )
        color = "green" if rank <= 3 else "orange" if rank <= 8 else "blue"
        folium.Marker(
            [candidate.lat, candidate.lon],
            tooltip=f"#{rank} · {candidate.final_score:.1f}",
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color=color, icon="home", prefix="fa"),
        ).add_to(results_map)
    st_folium(results_map, height=560, use_container_width=True, key="results_map")

    st.subheader("Kandidaten in volgorde")
    card_columns = st.columns(2)
    for rank, candidate in enumerate(result.candidates, start=1):
        with card_columns[(rank - 1) % 2]:
            with st.container(border=True):
                st.markdown(
                    f"### #{rank} · {candidate.candidate_id} — {candidate.final_score:.1f}/100"
                )
                if candidate.crop_path and Path(candidate.crop_path).exists():
                    st.image(
                        candidate.crop_path,
                        caption="Rood: kandidaatcontour · noord is boven",
                        width="stretch",
                    )
                score_parts = [f"lokaal {candidate.local_score:.1f}"]
                if candidate.luna_score is not None:
                    score_parts.append(f"Luna {candidate.luna_score:.1f}")
                if candidate.terra_score is not None:
                    score_parts.append(f"Terra {candidate.terra_score:.1f}")
                st.caption(" · ".join(score_parts))
                st.write(
                    f"**{candidate.area_m2:.0f} m²** · {LABELS.get(candidate.size_class, candidate.size_class)} · "
                    f"{LABELS.get(candidate.shape_class, candidate.shape_class)} · "
                    f"{candidate.buildings_within_50m} andere contour(en) binnen 50 m"
                )
                if candidate.reasons:
                    st.markdown(
                        "**Sterke punten**\n\n"
                        + "\n".join(f"- {item}" for item in candidate.reasons)
                    )
                if candidate.conflicts:
                    st.markdown(
                        "**Tegenstrijdigheden**\n\n"
                        + "\n".join(f"- {item}" for item in candidate.conflicts)
                    )
                if candidate.uncertainties:
                    st.markdown(
                        "**Onzekerheden**\n\n"
                        + "\n".join(f"- {item}" for item in candidate.uncertainties)
                    )
                maps_link = f"https://www.google.com/maps/search/?api=1&query={candidate.lat},{candidate.lon}"
                street_link = (
                    "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
                    f"{candidate.lat},{candidate.lon}"
                )
                st.markdown(f"[Google Maps]({maps_link}) · [Street View]({street_link})")

    if ledger.entries:
        with st.expander("Kostenlog"):
            st.dataframe(
                [
                    {
                        "stap": entry.label,
                        "model": entry.model,
                        "inputtokens": entry.input_tokens,
                        "outputtokens": entry.output_tokens,
                        "kosten_usd": round(entry.cost_usd, 6),
                        "geschat": entry.estimated,
                    }
                    for entry in ledger.entries
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "Dit is een kandidaat-rangschikking, geen bewijs van een exacte locatie. Controleer altijd meerdere "
        "aanwijzingen en houd rekening met het opnamejaar van de luchtfoto."
    )
