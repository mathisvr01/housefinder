from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from shapely.ops import transform
from urllib3.util.retry import Retry

from .config import Settings
from .geo import circle_bbox, haversine_m
from .models import Building


class IGNError(RuntimeError):
    pass


class JsonDiskCache:
    def __init__(self, directory: Path, ttl_seconds: int = 7 * 24 * 3600):
        self.directory = directory
        self.ttl_seconds = ttl_seconds
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict) -> None:
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def build_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "HouseFinder/2.0 (property-search prototype)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class IGNClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or build_http_session()
        self.cache = JsonDiskCache(settings.cache_dir / "wfs")
        self.to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

    def geocode(self, query: str) -> tuple[float, float, str] | None:
        query = query.strip()
        if not query:
            return None
        endpoints = (self.settings.geocode_url, self.settings.geocode_fallback_url)
        for endpoint in endpoints:
            try:
                response = self.session.get(
                    endpoint,
                    params={"q": query, "limit": 1},
                    timeout=(5, 15),
                )
                response.raise_for_status()
                data = response.json()
                features = data.get("features", [])
                if not features:
                    continue
                feature = features[0]
                lon, lat = feature["geometry"]["coordinates"][:2]
                label = feature.get("properties", {}).get("label", query)
                return float(lat), float(lon), str(label)
            except (requests.RequestException, ValueError, KeyError, TypeError):
                continue
        return None

    def fetch_buildings(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        progress: Callable[[str], None] | None = None,
    ) -> list[Building]:
        min_lon, min_lat, max_lon, max_lat = circle_bbox(center_lon, center_lat, radius_m)
        bbox = f"{min_lon:.7f},{min_lat:.7f},{max_lon:.7f},{max_lat:.7f},CRS:84"
        page_size = 1_000
        start_index = 0
        raw_features: list[dict] = []
        number_matched: int | None = None

        while start_index < self.settings.max_wfs_features:
            params = {
                "SERVICE": "WFS",
                "REQUEST": "GetFeature",
                "VERSION": "2.0.0",
                "TYPENAMES": self.settings.wfs_layer,
                "OUTPUTFORMAT": "application/json",
                "SRSNAME": "CRS:84",
                "COUNT": page_size,
                "STARTINDEX": start_index,
                "BBOX": bbox,
            }
            cache_key = json.dumps(params, sort_keys=True)
            payload = self.cache.get(cache_key)
            if payload is None:
                try:
                    response = self.session.get(
                        self.settings.wfs_url,
                        params=params,
                        timeout=(8, 35),
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError) as exc:
                    raise IGNError(
                        f"De openbare IGN-gebouwlaag kon niet worden opgehaald: {exc}"
                    ) from exc
                self.cache.put(cache_key, payload)

            features = payload.get("features", [])
            raw_features.extend(features)
            if number_matched is None:
                try:
                    number_matched = int(payload.get("numberMatched"))
                except (TypeError, ValueError):
                    number_matched = None
            if progress:
                total = number_matched if number_matched is not None else "?"
                progress(f"{len(raw_features)} van {total} gebouwcontouren opgehaald")
            if len(features) < page_size:
                break
            start_index += page_size

        if number_matched and number_matched > self.settings.max_wfs_features:
            raise IGNError(
                f"Het gebied bevat {number_matched} gebouwen; verklein de cirkel zodat maximaal "
                f"{self.settings.max_wfs_features} objecten nodig zijn."
            )

        buildings: list[Building] = []
        for feature in raw_features:
            geometry_data = feature.get("geometry")
            if not geometry_data:
                continue
            try:
                geometry_wgs84 = shape(geometry_data)
                if not geometry_wgs84.is_valid:
                    geometry_wgs84 = geometry_wgs84.buffer(0)
                geometry_l93 = transform(self.to_l93.transform, geometry_wgs84)
                if geometry_l93.is_empty:
                    continue
                centroid_l93 = geometry_l93.centroid
                centroid_wgs84 = geometry_wgs84.centroid
                area_m2 = float(geometry_l93.area)
            except Exception:
                continue

            if area_m2 < 4.0 or area_m2 > 20_000.0:
                continue
            lon = float(centroid_wgs84.x)
            lat = float(centroid_wgs84.y)
            if haversine_m(center_lon, center_lat, lon, lat) > radius_m:
                continue
            buildings.append(
                Building(
                    source_id=str(
                        feature.get("id", feature.get("properties", {}).get("cleabs", ""))
                    ),
                    geometry_wgs84=geometry_wgs84,
                    geometry_l93=geometry_l93,
                    lon=lon,
                    lat=lat,
                    x=float(centroid_l93.x),
                    y=float(centroid_l93.y),
                    area_m2=area_m2,
                    properties=feature.get("properties", {}),
                )
            )
        return buildings
