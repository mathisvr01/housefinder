from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.6-luna": ModelPricing(0.20, 1.20),
    "gpt-5.6-terra": ModelPricing(2.00, 12.00),
}


@dataclass(frozen=True)
class Settings:
    wfs_url: str = "https://data.geopf.fr/wfs/ows"
    wfs_layer: str = "BDTOPO_V3:batiment"
    wmts_url: str = "https://data.geopf.fr/wmts"
    wmts_layer: str = "ORTHOIMAGERY.ORTHOPHOTOS"
    geocode_url: str = "https://data.geopf.fr/geocodage/search"
    geocode_fallback_url: str = "https://api-adresse.data.gouv.fr/search/"
    luna_model: str = "gpt-5.6-luna"
    terra_model: str = "gpt-5.6-terra"
    max_photos: int = 6
    max_photo_side: int = 1024
    local_shortlist_size: int = 15
    terra_shortlist_size: int = 4
    displayed_candidates: int = 15
    imagery_zoom: int = 19
    imagery_crop_size: int = 512
    max_wfs_features: int = 8_000
    default_budget_usd: float = 0.04
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("HOUSEFINDER_CACHE_DIR", ".cache/housefinder"))
    )


DEFAULT_SETTINGS = Settings()
