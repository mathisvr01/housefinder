from dataclasses import replace

from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from housefinder.config import DEFAULT_SETTINGS
from housefinder.imagery import IGNImageryClient
from housefinder.models import Candidate


def test_candidate_outline_accepts_ign_xyz_coordinates(tmp_path):
    geometry_wgs84 = Polygon(
        [
            (1.8325, 44.8911, 210.0),
            (1.8328, 44.8911, 211.0),
            (1.8328, 44.8913, 212.0),
            (1.8325, 44.8913, 210.0),
            (1.8325, 44.8911, 210.0),
        ]
    )
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    geometry_l93 = transform(to_l93.transform, geometry_wgs84)
    candidate = Candidate(
        source_id="xyz-building",
        geometry_wgs84=geometry_wgs84,
        geometry_l93=geometry_l93,
        lon=1.83265,
        lat=44.8912,
        x=geometry_l93.centroid.x,
        y=geometry_l93.centroid.y,
        area_m2=geometry_l93.area,
        candidate_id="K01",
    )
    client = IGNImageryClient(replace(DEFAULT_SETTINGS, cache_dir=tmp_path))

    result = client._draw_candidate_outline(
        Image.new("RGB", (512, 512), "white"),
        candidate,
        zoom=19,
        left_world_px=67_790_000,
        top_world_px=47_000_000,
    )

    assert result.size == (512, 512)
    assert result.mode == "RGB"
