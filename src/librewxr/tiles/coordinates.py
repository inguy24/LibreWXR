# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math
from collections import OrderedDict, namedtuple
from threading import Lock

import numpy as np

from librewxr.config import settings
from librewxr.data.regions import REGIONS, RegionDef

# Legacy constants for USCOMP (kept for backward compatibility)
_USCOMP = REGIONS["USCOMP"]
WEST = _USCOMP.west
EAST = _USCOMP.east
NORTH = _USCOMP.north
SOUTH = _USCOMP.south
PIXEL_SIZE = _USCOMP.pixel_size
COMPOSITE_WIDTH = _USCOMP.width
COMPOSITE_HEIGHT = _USCOMP.height


# ── WGS84 ellipsoidal constants ────────────────────────────────────

_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2
_WGS84_E = math.sqrt(_WGS84_E2)

# ── Lambert Azimuthal Equal Area (LAEA) projection ────────────────────


def _laea_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 ellipsoidal Lambert Azimuthal Equal Area forward projection.

    Implements the oblique case per Snyder (1987) §24 / EPSG guidance
    note 7-2.  The projection parameters are taken from the RegionDef's
    ``laea_*`` fields (lat_0, lon_0, x_0, y_0).
    """
    phi_0 = math.radians(region.laea_lat0)
    lam_0 = math.radians(region.laea_lon0)

    # Eccentricity-derived constants at the origin latitude
    sin_phi0 = math.sin(phi_0)
    cos_phi0 = math.cos(phi_0)
    q_p = (1 - _WGS84_E2) * (
        1 / (1 - _WGS84_E2) - (1 / (2 * _WGS84_E)) * math.log((1 - _WGS84_E) / (1 + _WGS84_E))
    )
    q_0 = _laea_q(sin_phi0)
    beta_0 = math.asin(q_0 / q_p)
    R_q = _WGS84_A * math.sqrt(q_p / 2)
    D = _WGS84_A * cos_phi0 / (
        math.sqrt(1 - _WGS84_E2 * sin_phi0 ** 2) * R_q * math.cos(beta_0)
    )

    # Per-point computations (vectorized)
    phi = np.radians(lat)
    lam = np.radians(lon)
    sin_phi = np.sin(phi)
    q = _laea_q_vec(sin_phi)
    beta = np.arcsin(np.clip(q / q_p, -1.0, 1.0))

    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    lam_diff = lam - lam_0

    B = R_q * np.sqrt(
        2.0 / (
            1
            + math.sin(beta_0) * sin_beta
            + math.cos(beta_0) * cos_beta * np.cos(lam_diff)
        )
    )

    x = B * D * cos_beta * np.sin(lam_diff) + region.laea_x0
    y = (B / D) * (
        math.cos(beta_0) * sin_beta
        - math.sin(beta_0) * cos_beta * np.cos(lam_diff)
    ) + region.laea_y0

    return x, y


def _laea_q(sin_phi: float) -> float:
    """Authalic latitude helper q (scalar)."""
    return (1 - _WGS84_E2) * (
        sin_phi / (1 - _WGS84_E2 * sin_phi ** 2)
        - (1 / (2 * _WGS84_E)) * math.log(
            (1 - _WGS84_E * sin_phi) / (1 + _WGS84_E * sin_phi)
        )
    )


def _laea_q_vec(sin_phi: np.ndarray) -> np.ndarray:
    """Authalic latitude helper q (vectorized)."""
    return (1 - _WGS84_E2) * (
        sin_phi / (1 - _WGS84_E2 * sin_phi ** 2)
        - (1 / (2 * _WGS84_E)) * np.log(
            (1 - _WGS84_E * sin_phi) / (1 + _WGS84_E * sin_phi)
        )
    )


def _laea_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for a LAEA region."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _laea_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Transverse Mercator projection (spherical) ────────────────────────
# Snyder (1987) §8 — used by composites that specify their grid on a
# sphere rather than an ellipsoid (DPC Italy: R=6371229, lat_0=42°,
# lon_0=12.5°).  For met composites that stay well inside ±1000 km of
# the central meridian, the spherical form is accurate to a few metres
# vs. the WGS84 ellipsoidal form — well below 1 km/pixel resolution.


def _tmerc_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical Transverse Mercator forward projection.

    Reads parameters from the RegionDef's ``tmerc_*`` fields.  The
    formulas have singularities at λ - λ₀ = ±90°; those are far outside
    every meteorological domain we care about — Italy's λ range stays
    within ±8° of λ₀=12.5°.
    """
    phi_0 = math.radians(region.tmerc_lat0)
    lam_0 = math.radians(region.tmerc_lon0)
    R = region.tmerc_radius
    k0 = region.tmerc_k0

    phi = np.radians(lat)
    lam = np.radians(lon)
    lam_diff = lam - lam_0

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_dlam = np.cos(lam_diff)
    sin_dlam = np.sin(lam_diff)

    # x = R · k0 · arctanh(cos(φ) · sin(λ - λ₀))
    # Clip just inside ±1 to keep arctanh finite at the antipodal cusp.
    B = cos_phi * sin_dlam
    B = np.clip(B, -0.9999999, 0.9999999)
    x = R * k0 * np.arctanh(B)

    # y = R · k0 · (atan2(sin(φ), cos(φ) · cos(λ-λ₀)) - φ₀) — quadrant-safe
    # equivalent of the textbook arctan(tan(φ) / cos(λ-λ₀)) form.
    y = R * k0 * (np.arctan2(sin_phi, cos_phi * cos_dlam) - phi_0)

    return x, y


def _tmerc_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for a tmerc region."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _tmerc_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Region-aware coordinate functions ────────────────────────────────

_CacheInfo = namedtuple("_CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def _sized_lru_cache(maxsize: int):
    """Thread-safe LRU memoizer that measures actual result-array bytes.

    Drop-in for ``functools.lru_cache`` within this module: exposes the
    same ``cache_info()`` / ``cache_clear()`` surface, plus
    ``cache_bytes()`` returning the measured size of resident entries.
    Exists because entry sizes here vary 4x with the ``tile_size``
    argument (256 px ≈ 0.5 MB, 512 px ≈ 2 MB) — a count-based estimate
    misattributed ~134 MB of 512 px satellite entries to "other" in the
    /health breakdown (audit 2026-08-08).
    """

    def decorator(fn):
        cache: OrderedDict[tuple, tuple] = OrderedDict()
        lock = Lock()
        stats = {"hits": 0, "misses": 0, "bytes": 0}

        def _entry_bytes(result: tuple) -> int:
            return sum(int(a.nbytes) for a in result)

        def wrapper(*args):
            with lock:
                if args in cache:
                    cache.move_to_end(args)
                    stats["hits"] += 1
                    return cache[args]
                stats["misses"] += 1
            result = fn(*args)
            with lock:
                if args not in cache:
                    cache[args] = result
                    stats["bytes"] += _entry_bytes(result)
                    while len(cache) > maxsize:
                        _, evicted = cache.popitem(last=False)
                        stats["bytes"] -= _entry_bytes(evicted)
            return result

        def cache_info() -> _CacheInfo:
            with lock:
                return _CacheInfo(
                    stats["hits"], stats["misses"], maxsize, len(cache)
                )

        def cache_clear() -> None:
            with lock:
                cache.clear()
                stats["hits"] = 0
                stats["misses"] = 0
                stats["bytes"] = 0

        def cache_bytes() -> int:
            with lock:
                return stats["bytes"]

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper.cache_bytes = cache_bytes
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute composite pixel indices for a tile within a specific region.

    Returns (row_indices, col_indices) arrays of shape (tile_size, tile_size).
    Values of -1 indicate pixels outside the region's coverage.
    """
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    col_idx = np.rint(col_grid).astype(np.int32)
    row_idx = np.rint(row_grid).astype(np.int32)

    oob = (
        (col_idx < 0)
        | (col_idx >= region.width)
        | (row_idx < 0)
        | (row_idx >= region.height)
    )
    col_idx[oob] = -1
    row_idx[oob] = -1

    col_idx.flags.writeable = False
    row_idx.flags.writeable = False
    return row_idx, col_idx


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute composite pixel indices for a tile with padding within a region."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    col_idx = np.rint(col_grid).astype(np.int32)
    row_idx = np.rint(row_grid).astype(np.int32)

    oob = (
        (col_idx < 0)
        | (col_idx >= region.width)
        | (row_idx < 0)
        | (row_idx >= region.height)
    )
    col_idx[oob] = -1
    row_idx[oob] = -1

    col_idx.flags.writeable = False
    row_idx.flags.writeable = False
    return row_idx, col_idx


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_fractional(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fractional composite pixel coordinates for bilinear interpolation."""
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    row_grid = np.clip(row_grid, 0, region.height - 1).astype(np.float32)
    col_grid = np.clip(col_grid, 0, region.width - 1).astype(np.float32)

    row_grid.flags.writeable = False
    col_grid.flags.writeable = False
    return row_grid, col_grid


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_fractional_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional pixel coords for a tile with padding (bilinear + blur path)."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    row_grid = np.clip(row_grid, 0, region.height - 1).astype(np.float32)
    col_grid = np.clip(col_grid, 0, region.width - 1).astype(np.float32)

    row_grid.flags.writeable = False
    col_grid.flags.writeable = False
    return row_grid, col_grid


def tile_overlaps_region(region: RegionDef, z: int, x: int, y: int) -> bool:
    """Check if a tile has any overlap with a region's coverage area."""
    tw, ts, te, tn = tile_bounds(z, x, y)
    return not (
        te < region.west or tw > region.east
        or tn < region.south or ts > region.north
    )


def overlapping_regions(
    z: int, x: int, y: int, enabled: list[str] | None = None
) -> list[RegionDef]:
    """Return list of regions that overlap a given tile.

    Sorted by pixel_size ascending (finest resolution first).
    """
    if enabled is None:
        enabled = list(REGIONS.keys())

    result = []
    for name in enabled:
        region = REGIONS.get(name)
        if region and tile_overlaps_region(region, z, x, y):
            result.append(region)

    # Finest resolution first (smallest pixel_size)
    result.sort(key=lambda r: r.pixel_size)
    return result


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_latlons(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon for each pixel in a Web Mercator tile.

    Returns (lat_grid, lon_grid) float32 arrays of shape (tile_size, tile_size).
    Used for temperature lookups that need geographic coordinates.
    float32 provides ~7 decimal digits (~0.00001° ≈ 1 m precision),
    far exceeding any radar data resolution.
    """
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float32) + 0.5
    cy = np.arange(tile_size, dtype=np.float32) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.float32(math.pi) * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_grid.flags.writeable = False
    lat_grid.flags.writeable = False
    return lat_grid, lon_grid


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_latlons_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon for a tile with padding."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float32) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float32) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.float32(math.pi) * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_grid.flags.writeable = False
    lat_grid.flags.writeable = False
    return lat_grid, lon_grid


# ── Legacy USCOMP-only functions (kept for backward compatibility) ───


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for a tile."""
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices for a tile (legacy wrapper)."""
    return region_pixel_indices(_USCOMP, z, x, y, tile_size)


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices with padding (legacy wrapper)."""
    return region_pixel_indices_padded(_USCOMP, z, x, y, tile_size, pad)


@_sized_lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices_fractional(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP fractional indices (legacy wrapper)."""
    return region_pixel_indices_fractional(_USCOMP, z, x, y, tile_size)


def tile_overlaps_composite(z: int, x: int, y: int) -> bool:
    """Check if a tile overlaps USCOMP (legacy wrapper)."""
    return tile_overlaps_region(_USCOMP, z, x, y)


# ---------------------------------------------------------------------------
# Cache pre-warming
# ---------------------------------------------------------------------------


def warm_coordinate_caches(
    enabled_regions: list[str] | None, max_zoom: int, tile_size: int = 256
) -> int:
    """Pre-populate all coordinate LRU caches up to ``max_zoom``.

    Iterates every tile coordinate at zooms 0 through ``max_zoom``,
    computes overlapping regions, and calls each cached coordinate
    function so that real tile requests never pay the cold-start cost
    of trigonometric projections and array allocations.

    Returns the number of unique (region, z, x, y, tile_size) cache
    entries warmed.
    """
    if max_zoom <= 0:
        return 0
    warmed = 0
    for z in range(max_zoom + 1):
        n = 2**z
        for y in range(n):
            for x in range(n):
                regions = overlapping_regions(z, x, y, enabled_regions)
                if not regions:
                    continue
                # Tile-level lat/lon grids (used by ECMWF fallback, arrows)
                # Positional args only: the render path calls these
                # positionally, and cache keys are argument-shape
                # sensitive — a pad=8 kwarg here would warm entries the
                # render path could never hit (true under functools.
                # lru_cache as well; latent since the warmer was added).
                tile_pixel_latlons(z, x, y, tile_size)
                tile_pixel_latlons_padded(z, x, y, tile_size, 8)
                for region in regions:
                    region_pixel_indices(region, z, x, y, tile_size)
                    region_pixel_indices_padded(region, z, x, y, tile_size, 8)
                    region_pixel_indices_fractional(region, z, x, y, tile_size)
                    region_pixel_indices_fractional_padded(region, z, x, y, tile_size, 8)
                    warmed += 1
    return warmed


# All decorated coordinate cache functions (for bulk clear / size queries).
# Legacy wrappers (tile_pixel_indices, etc.) are excluded because they
# delegate to the corresponding region_pixel_* function and thus share
# the same underlying numpy arrays — counting them would double-count.
ALL_CACHES = [
    region_pixel_indices,
    region_pixel_indices_padded,
    region_pixel_indices_fractional,
    region_pixel_indices_fractional_padded,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
]

def coord_cache_stats() -> dict:
    """Per-cache hit/miss/fill stats for the /health endpoint.

    Hit ratio + fill ratio are what you want when tuning
    ``LIBREWXR_COORD_CACHE_SIZE``: low hit ratio with full caches means
    the cap is too small; full caches with high hit ratio means it's
    well-sized; partial fills mean the cap has headroom.
    """
    caches: dict[str, dict] = {}
    max_size = 0
    for fn in ALL_CACHES:
        info = fn.cache_info()
        max_size = info.maxsize or 0
        total = info.hits + info.misses
        caches[fn.__name__] = {
            "entries": info.currsize,
            "hits": info.hits,
            "misses": info.misses,
            "hit_ratio": round(info.hits / total, 3) if total else None,
        }
    return {"max_size": max_size, "caches": caches}


def coord_cache_bytes() -> int:
    """Total memory consumed by all coordinate LRU caches, measured.

    Sums each cache's ``cache_bytes()`` — the actual ``nbytes`` of every
    resident entry, recorded at insert/evict time. Replaces the old
    count × fixed-size estimate, which assumed 256 px entries and
    under-reported 512 px satellite entries 4x.
    """
    return sum(fn.cache_bytes() for fn in ALL_CACHES)
