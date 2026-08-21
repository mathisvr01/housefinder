from __future__ import annotations

import math
import re
from urllib.parse import unquote

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def circle_bbox(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111_320.0
    lon_scale = max(0.05, math.cos(math.radians(lat)))
    lon_delta = radius_m / (111_320.0 * lon_scale)
    return lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta


def extract_coords_from_url(url: str) -> tuple[float, float] | None:
    value = unquote(url.strip())

    bienici = re.search(r"camera=\d+(?:\.\d+)?_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", value)
    if bienici:
        longitude = float(bienici.group(1))
        latitude = float(bienici.group(2))
        if valid_lon_lat(longitude, latitude):
            return latitude, longitude

    google = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", value)
    if google:
        latitude = float(google.group(1))
        longitude = float(google.group(2))
        if valid_lon_lat(longitude, latitude):
            return latitude, longitude

    query = re.search(r"(?:query|q)=(-?\d+(?:\.\d+)?)[,%20 ]+(-?\d+(?:\.\d+)?)", value)
    if query:
        latitude = float(query.group(1))
        longitude = float(query.group(2))
        if valid_lon_lat(longitude, latitude):
            return latitude, longitude

    return None


def valid_lon_lat(lon: float, lat: float) -> bool:
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


def lonlat_to_world_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = min(85.05112878, max(-85.05112878, lat))
    scale = 256.0 * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def world_pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = 256.0 * (2**zoom)
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat
