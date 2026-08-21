from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from shapely.geometry.base import BaseGeometry

SizeCategory = Literal["small", "medium", "large", "unknown"]
FootprintShape = Literal[
    "compact", "elongated", "complex", "l_shape", "u_shape", "courtyard", "unknown"
]
SettingCategory = Literal["isolated", "small_cluster", "village", "unknown"]
VegetationCategory = Literal["open", "mixed", "wooded", "unknown"]
RoadContext = Literal["roadside", "short_drive", "long_drive", "unknown"]
Ternary = Literal["yes", "no", "unknown"]


class ListingProfile(BaseModel):
    """Only clues that can reasonably help when inspecting an aerial image."""

    model_config = ConfigDict(extra="forbid")

    size_category: SizeCategory
    size_confidence: float = Field(ge=0.0, le=1.0)
    footprint_shape: FootprintShape
    shape_confidence: float = Field(ge=0.0, le=1.0)
    outbuildings_visible: int = Field(ge=0, le=9)
    outbuildings_confidence: float = Field(ge=0.0, le=1.0)
    absence_of_outbuildings_conclusive: bool
    setting: SettingCategory
    setting_confidence: float = Field(ge=0.0, le=1.0)
    vegetation: VegetationCategory
    vegetation_confidence: float = Field(ge=0.0, le=1.0)
    road_context: RoadContext
    road_confidence: float = Field(ge=0.0, le=1.0)
    pool_visible: Ternary
    pool_confidence: float = Field(ge=0.0, le=1.0)
    roof_color: str = Field(max_length=80)
    roof_color_confidence: float = Field(ge=0.0, le=1.0)
    diagnostic_photo_indices: list[int] = Field(max_length=4)
    summary: str = Field(max_length=500)

    @field_validator("diagnostic_photo_indices")
    @classmethod
    def unique_photo_indices(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(index for index in value if index > 0))[:4]

    @classmethod
    def unknown(cls) -> ListingProfile:
        return cls(
            size_category="unknown",
            size_confidence=0.0,
            footprint_shape="unknown",
            shape_confidence=0.0,
            outbuildings_visible=0,
            outbuildings_confidence=0.0,
            absence_of_outbuildings_conclusive=False,
            setting="unknown",
            setting_confidence=0.0,
            vegetation="unknown",
            vegetation_confidence=0.0,
            road_context="unknown",
            road_confidence=0.0,
            pool_visible="unknown",
            pool_confidence=0.0,
            roof_color="unknown",
            roof_color_confidence=0.0,
            diagnostic_photo_indices=[],
            summary="Nog geen fotoanalyse uitgevoerd; alleen lokale geometrische ranking.",
        )


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    visual_score: float = Field(ge=0.0, le=100.0)
    supporting_clues: list[str] = Field(max_length=5)
    conflicts: list[str] = Field(max_length=5)
    uncertainties: list[str] = Field(max_length=5)


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[CandidateAssessment]
    summary: str = Field(max_length=500)


@dataclass
class Building:
    source_id: str
    geometry_wgs84: BaseGeometry
    geometry_l93: BaseGeometry
    lon: float
    lat: float
    x: float
    y: float
    area_m2: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate(Building):
    candidate_id: str = ""
    elongation: float = 1.0
    fill_ratio: float = 1.0
    compactness: float = 1.0
    shape_class: str = "compact"
    size_class: str = "medium"
    setting_class: str = "unknown"
    buildings_within_20m: int = 0
    buildings_within_50m: int = 0
    buildings_within_100m: int = 0
    nearest_building_m: float = 9999.0
    distance_to_center_m: float = 0.0
    local_score: float = 0.0
    luna_score: float | None = None
    terra_score: float | None = None
    final_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    crop_path: str | None = None


@dataclass
class SearchResult:
    candidates: list[Candidate]
    total_buildings: int
    shortlist_size: int
    contact_sheet_path: str | None
    warnings: list[str] = field(default_factory=list)
