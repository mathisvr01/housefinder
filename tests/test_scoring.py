from __future__ import annotations

from shapely.geometry import box

from housefinder.models import Candidate, ListingProfile
from housefinder.scoring import geometry_features, score_candidate


def make_candidate(area: float, size_class: str, setting: str, nearby: int) -> Candidate:
    geometry = box(0, 0, 20, max(1, area / 20))
    return Candidate(
        source_id="building.1",
        geometry_wgs84=box(1.83, 44.89, 1.8301, 44.8901),
        geometry_l93=geometry,
        lon=1.83,
        lat=44.89,
        x=0,
        y=0,
        area_m2=area,
        candidate_id="C001",
        size_class=size_class,
        setting_class=setting,
        shape_class="compact",
        buildings_within_50m=nearby,
        distance_to_center_m=100,
    )


def test_geometry_features_distinguish_elongated_building() -> None:
    elongation, _, _, shape = geometry_features(box(0, 0, 40, 8))
    assert elongation == 5
    assert shape == "elongated"


def test_confirmed_profile_rewards_matching_candidate() -> None:
    profile = ListingProfile.unknown().model_copy(
        update={
            "size_category": "medium",
            "size_confidence": 0.95,
            "footprint_shape": "compact",
            "shape_confidence": 0.9,
            "setting": "isolated",
            "setting_confidence": 0.9,
        }
    )
    matching = make_candidate(150, "medium", "isolated", 1)
    mismatch = make_candidate(500, "large", "village", 8)
    match_score, _, _ = score_candidate(matching, profile, 600)
    mismatch_score, _, conflicts = score_candidate(mismatch, profile, 600)
    assert match_score > mismatch_score
    assert conflicts


def test_unconfirmed_absence_does_not_penalize_outbuildings() -> None:
    profile = ListingProfile.unknown().model_copy(
        update={
            "outbuildings_visible": 0,
            "outbuildings_confidence": 0.95,
            "absence_of_outbuildings_conclusive": False,
        }
    )
    candidate = make_candidate(150, "medium", "isolated", 5)
    _, _, conflicts = score_candidate(candidate, profile, 600)
    assert not any("gebouwen dichtbij" in item for item in conflicts)
