from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from .geo import haversine_m
from .models import Building, Candidate, ListingProfile


def build_and_score_candidates(
    buildings: list[Building],
    profile: ListingProfile,
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> list[Candidate]:
    if not buildings:
        return []

    all_x = np.asarray([building.x for building in buildings], dtype=float)
    all_y = np.asarray([building.y for building in buildings], dtype=float)
    candidates: list[Candidate] = []

    for building in buildings:
        if building.area_m2 < 20.0 or building.area_m2 > 5_000.0:
            continue
        distances = np.hypot(all_x - building.x, all_y - building.y)
        other = distances > 0.25
        within_20 = int(np.count_nonzero(other & (distances <= 20.0)))
        within_50 = int(np.count_nonzero(other & (distances <= 50.0)))
        within_100 = int(np.count_nonzero(other & (distances <= 100.0)))
        nearest = float(np.min(distances[other])) if np.any(other) else 9_999.0

        elongation, fill_ratio, compactness, shape_class = geometry_features(building.geometry_l93)
        size_class = classify_size(building.area_m2)
        setting_class = classify_setting(within_100)
        distance_to_center = haversine_m(center_lon, center_lat, building.lon, building.lat)

        candidate = Candidate(
            source_id=building.source_id,
            geometry_wgs84=building.geometry_wgs84,
            geometry_l93=building.geometry_l93,
            lon=building.lon,
            lat=building.lat,
            x=building.x,
            y=building.y,
            area_m2=building.area_m2,
            properties=building.properties,
            elongation=elongation,
            fill_ratio=fill_ratio,
            compactness=compactness,
            shape_class=shape_class,
            size_class=size_class,
            setting_class=setting_class,
            buildings_within_20m=within_20,
            buildings_within_50m=within_50,
            buildings_within_100m=within_100,
            nearest_building_m=nearest,
            distance_to_center_m=distance_to_center,
        )
        score, reasons, conflicts = score_candidate(candidate, profile, radius_m)
        candidate.local_score = score
        candidate.final_score = score
        candidate.reasons = reasons
        candidate.conflicts = conflicts
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate.local_score, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = f"C{rank:03d}"
    return candidates


def geometry_features(geometry) -> tuple[float, float, float, str]:
    rectangle = geometry.minimum_rotated_rectangle
    rectangle_area = max(float(rectangle.area), 1e-6)
    coordinates = list(rectangle.exterior.coords)
    edge_lengths = [
        math.hypot(
            coordinates[i + 1][0] - coordinates[i][0],
            coordinates[i + 1][1] - coordinates[i][1],
        )
        for i in range(4)
    ]
    long_side = max(edge_lengths)
    short_side = max(min(edge_lengths), 1e-6)
    elongation = long_side / short_side
    fill_ratio = min(1.0, float(geometry.area) / rectangle_area)
    perimeter = max(float(geometry.length), 1e-6)
    compactness = min(1.0, 4.0 * math.pi * float(geometry.area) / (perimeter**2))
    vertices = exterior_vertex_count(geometry)

    if elongation >= 2.2:
        shape_class = "elongated"
    elif fill_ratio < 0.72 or vertices >= 9:
        shape_class = "complex"
    else:
        shape_class = "compact"
    return elongation, fill_ratio, compactness, shape_class


def exterior_vertex_count(geometry) -> int:
    polygon: Polygon | None
    if isinstance(geometry, Polygon):
        polygon = geometry
    elif isinstance(geometry, MultiPolygon) and geometry.geoms:
        polygon = max(geometry.geoms, key=lambda item: item.area)
    else:
        polygon = None
    return max(0, len(polygon.exterior.coords) - 1) if polygon else 0


def classify_size(area_m2: float) -> str:
    if area_m2 < 90.0:
        return "small"
    if area_m2 <= 240.0:
        return "medium"
    return "large"


def classify_setting(buildings_within_100m: int) -> str:
    if buildings_within_100m <= 2:
        return "isolated"
    if buildings_within_100m <= 10:
        return "small_cluster"
    return "village"


def score_candidate(
    candidate: Candidate, profile: ListingProfile, radius_m: float
) -> tuple[float, list[str], list[str]]:
    weighted: list[tuple[float, float]] = []
    reasons: list[str] = []
    conflicts: list[str] = []

    if profile.size_category != "unknown" and profile.size_confidence >= 0.25:
        match = 1.0 if candidate.size_class == profile.size_category else 0.15
        weighted.append((0.28 * profile.size_confidence, match))
        if match > 0.8:
            reasons.append(
                f"Voetafdruk {candidate.area_m2:.0f} m² past bij categorie {profile.size_category}."
            )
        else:
            conflicts.append(
                f"Voetafdruk is {candidate.size_class}, verwacht {profile.size_category}."
            )

    if profile.footprint_shape != "unknown" and profile.shape_confidence >= 0.25:
        expected = normalize_shape(profile.footprint_shape)
        match = 1.0 if candidate.shape_class == expected else 0.25
        weighted.append((0.20 * profile.shape_confidence, match))
        if match > 0.8:
            reasons.append(f"Gebouwvorm is {candidate.shape_class}.")
        else:
            conflicts.append(
                f"Gebouwvorm is {candidate.shape_class}, verwacht {profile.footprint_shape}."
            )

    use_outbuildings = profile.outbuildings_confidence >= 0.35 and (
        profile.outbuildings_visible > 0 or profile.absence_of_outbuildings_conclusive
    )
    if use_outbuildings:
        difference = abs(candidate.buildings_within_50m - profile.outbuildings_visible)
        match = max(0.0, 1.0 - difference / 4.0)
        weighted.append((0.20 * profile.outbuildings_confidence, match))
        if difference <= 1:
            reasons.append(
                f"{candidate.buildings_within_50m} andere gebouwcontour(en) binnen 50 m."
            )
        else:
            conflicts.append(
                f"{candidate.buildings_within_50m} gebouwen dichtbij versus "
                f"{profile.outbuildings_visible} zichtbaar op de foto's."
            )

    if profile.setting != "unknown" and profile.setting_confidence >= 0.25:
        match = 1.0 if candidate.setting_class == profile.setting else 0.25
        weighted.append((0.24 * profile.setting_confidence, match))
        if match > 0.8:
            reasons.append(f"Omgevingsdichtheid past bij {profile.setting}.")
        else:
            conflicts.append(
                f"Omgevingsdichtheid is {candidate.setting_class}, verwacht {profile.setting}."
            )

    # A weak center prior breaks ties but never excludes the edge of the circle.
    center_prior = max(0.0, 1.0 - candidate.distance_to_center_m / max(radius_m, 1.0))
    weighted.append((0.08, center_prior))

    total_weight = sum(weight for weight, _ in weighted)
    score = 100.0 * sum(weight * value for weight, value in weighted) / total_weight
    return round(score, 2), reasons[:5], conflicts[:5]


def normalize_shape(shape: str) -> str:
    if shape in {"l_shape", "u_shape", "courtyard"}:
        return "complex"
    return shape


def apply_luna_scores(candidates: list[Candidate], scores: dict[str, float]) -> None:
    for candidate in candidates:
        if candidate.candidate_id in scores:
            candidate.luna_score = scores[candidate.candidate_id]
            candidate.final_score = round(
                0.55 * candidate.local_score + 0.45 * candidate.luna_score, 2
            )


def apply_terra_scores(candidates: list[Candidate], scores: dict[str, float]) -> None:
    for candidate in candidates:
        if candidate.candidate_id in scores:
            candidate.terra_score = scores[candidate.candidate_id]
            luna = (
                candidate.luna_score if candidate.luna_score is not None else candidate.local_score
            )
            candidate.final_score = round(
                0.25 * candidate.local_score + 0.15 * luna + 0.60 * candidate.terra_score,
                2,
            )


def is_ambiguous(candidates: list[Candidate]) -> bool:
    ranked = sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)
    if len(ranked) < 2:
        return False
    return ranked[0].final_score < 72.0 or ranked[0].final_score - ranked[1].final_score < 7.0
