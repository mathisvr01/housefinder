from __future__ import annotations

import pytest

from housefinder.geo import (
    circle_bbox,
    extract_coords_from_url,
    haversine_m,
    lonlat_to_world_pixel,
    world_pixel_to_lonlat,
)


def test_extract_bienici_coordinates() -> None:
    assert extract_coords_from_url("https://example.test/?camera=14_1.832689_44.891237") == (
        44.891237,
        1.832689,
    )


def test_extract_google_coordinates() -> None:
    assert extract_coords_from_url("https://maps.google.com/@44.89,1.83,15z") == (
        44.89,
        1.83,
    )


def test_circle_bbox_contains_cardinal_points() -> None:
    min_lon, min_lat, max_lon, max_lat = circle_bbox(1.83, 44.89, 600)
    assert min_lon < 1.83 < max_lon
    assert min_lat < 44.89 < max_lat
    assert haversine_m(1.83, 44.89, 1.83, max_lat) == pytest.approx(600, rel=0.01)


def test_world_pixel_roundtrip() -> None:
    x, y = lonlat_to_world_pixel(1.832689, 44.891237, 19)
    lon, lat = world_pixel_to_lonlat(x, y, 19)
    assert lon == pytest.approx(1.832689, abs=1e-8)
    assert lat == pytest.approx(44.891237, abs=1e-8)
