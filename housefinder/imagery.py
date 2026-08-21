from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely.geometry import MultiPolygon, Polygon

from .config import Settings
from .geo import lonlat_to_world_pixel
from .ign import build_http_session
from .models import Candidate


class ImageryError(RuntimeError):
    pass


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class IGNImageryClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or build_http_session()
        self.tile_dir = settings.cache_dir / "tiles"
        self.crop_dir = settings.cache_dir / "crops"
        self.sheet_dir = settings.cache_dir / "contact_sheets"
        for directory in (self.tile_dir, self.crop_dir, self.sheet_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def tile_path(self, zoom: int, x: int, y: int) -> Path:
        return self.tile_dir / str(zoom) / str(x) / f"{y}.jpg"

    def fetch_tile(self, zoom: int, x: int, y: int) -> Path:
        path = self.tile_path(zoom, x, y)
        if path.exists() and path.stat().st_size > 500:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "SERVICE": "WMTS",
            "REQUEST": "GetTile",
            "VERSION": "1.0.0",
            "LAYER": self.settings.wmts_layer,
            "STYLE": "normal",
            "FORMAT": "image/jpeg",
            "TILEMATRIXSET": "PM",
            "TILEMATRIX": zoom,
            "TILEROW": y,
            "TILECOL": x,
        }
        try:
            response = self.session.get(
                self.settings.wmts_url,
                params=params,
                timeout=(7, 25),
            )
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            temporary = path.with_suffix(".tmp")
            image.save(temporary, format="JPEG", quality=92)
            temporary.replace(path)
            return path
        except (requests.RequestException, OSError) as exc:
            raise ImageryError(f"IGN-luchtfototegel {zoom}/{x}/{y} mislukt: {exc}") from exc

    def required_tiles(
        self, lon: float, lat: float, zoom: int, size_px: int
    ) -> set[tuple[int, int, int]]:
        center_x, center_y = lonlat_to_world_pixel(lon, lat, zoom)
        half = size_px / 2
        min_x = math.floor((center_x - half) / 256)
        max_x = math.floor((center_x + half - 1) / 256)
        min_y = math.floor((center_y - half) / 256)
        max_y = math.floor((center_y + half - 1) / 256)
        return {
            (zoom, tile_x, tile_y)
            for tile_x in range(min_x, max_x + 1)
            for tile_y in range(min_y, max_y + 1)
        }

    def prefetch_for_candidates(
        self,
        candidates: Iterable[Candidate],
        zoom: int,
        size_px: int,
        workers: int = 6,
    ) -> list[str]:
        tiles: set[tuple[int, int, int]] = set()
        for candidate in candidates:
            tiles.update(self.required_tiles(candidate.lon, candidate.lat, zoom, size_px))
        warnings: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.fetch_tile, z, x, y): (z, x, y) for z, x, y in sorted(tiles)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except ImageryError as exc:
                    warnings.append(str(exc))
        return warnings

    def candidate_crop(
        self,
        candidate: Candidate,
        zoom: int | None = None,
        size_px: int | None = None,
    ) -> tuple[Image.Image, Path]:
        zoom = zoom or self.settings.imagery_zoom
        size_px = size_px or self.settings.imagery_crop_size
        cache_id = hashlib.sha256(
            f"{candidate.source_id}|{zoom}|{size_px}|v2".encode()
        ).hexdigest()[:24]
        path = self.crop_dir / f"{cache_id}.jpg"
        if path.exists() and path.stat().st_size > 1_000:
            return Image.open(path).convert("RGB"), path

        center_x, center_y = lonlat_to_world_pixel(candidate.lon, candidate.lat, zoom)
        left = math.floor(center_x - size_px / 2)
        top = math.floor(center_y - size_px / 2)
        right = left + size_px
        bottom = top + size_px
        min_tile_x = math.floor(left / 256)
        max_tile_x = math.floor((right - 1) / 256)
        min_tile_y = math.floor(top / 256)
        max_tile_y = math.floor((bottom - 1) / 256)
        mosaic = Image.new(
            "RGB",
            ((max_tile_x - min_tile_x + 1) * 256, (max_tile_y - min_tile_y + 1) * 256),
        )
        for tile_x in range(min_tile_x, max_tile_x + 1):
            for tile_y in range(min_tile_y, max_tile_y + 1):
                tile_path = self.fetch_tile(zoom, tile_x, tile_y)
                with Image.open(tile_path) as tile:
                    mosaic.paste(
                        tile.convert("RGB"),
                        ((tile_x - min_tile_x) * 256, (tile_y - min_tile_y) * 256),
                    )

        origin_x = min_tile_x * 256
        origin_y = min_tile_y * 256
        crop = mosaic.crop((left - origin_x, top - origin_y, right - origin_x, bottom - origin_y))
        crop = self._draw_candidate_outline(crop, candidate, zoom, left, top)
        temporary = path.with_suffix(".tmp")
        crop.save(temporary, format="JPEG", quality=94)
        temporary.replace(path)
        return crop, path

    def _draw_candidate_outline(
        self,
        image: Image.Image,
        candidate: Candidate,
        zoom: int,
        left_world_px: int,
        top_world_px: int,
    ) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        geometry = candidate.geometry_wgs84
        polygons: list[Polygon] = []
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        for polygon in polygons:
            points = []
            for coordinate in polygon.exterior.coords:
                lon, lat = coordinate[:2]
                world_x, world_y = lonlat_to_world_pixel(lon, lat, zoom)
                points.append((world_x - left_world_px, world_y - top_world_px))
            if len(points) >= 3:
                draw.polygon(points, fill=(239, 68, 68, 45), outline=(255, 35, 35, 255), width=4)
        center = (image.width // 2, image.height // 2)
        draw.line(
            (center[0] - 8, center[1], center[0] + 8, center[1]), fill=(255, 255, 255, 230), width=2
        )
        draw.line(
            (center[0], center[1] - 8, center[0], center[1] + 8), fill=(255, 255, 255, 230), width=2
        )
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def contact_sheet(
        self,
        candidates: list[Candidate],
        crops: dict[str, Image.Image],
        *,
        columns: int = 3,
    ) -> tuple[Image.Image, Path]:
        cell_width = 360
        cell_height = 390
        header_height = 62
        rows = math.ceil(len(candidates) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#f1f5f9")
        draw = ImageDraw.Draw(sheet)
        title_font = load_font(21, bold=True)
        body_font = load_font(14)

        for index, candidate in enumerate(candidates):
            column = index % columns
            row = index // columns
            x = column * cell_width
            y = row * cell_height
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1), outline="#94a3b8", width=2
            )
            draw.text(
                (x + 12, y + 8),
                f"{candidate.candidate_id}  lokaal {candidate.local_score:.0f}/100",
                fill="#0f172a",
                font=title_font,
            )
            draw.text(
                (x + 12, y + 36),
                f"{candidate.area_m2:.0f} m² · {candidate.shape_class} · {candidate.buildings_within_50m} nabij · N ↑",
                fill="#334155",
                font=body_font,
            )
            crop = ImageOps.fit(
                crops[candidate.candidate_id], (cell_width - 16, cell_height - header_height - 12)
            )
            sheet.paste(crop, (x + 8, y + header_height))

        signature = "|".join(candidate.source_id for candidate in candidates)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
        path = self.sheet_dir / f"{digest}.jpg"
        temporary = path.with_suffix(".tmp")
        sheet.save(temporary, format="JPEG", quality=91)
        temporary.replace(path)
        return sheet, path
