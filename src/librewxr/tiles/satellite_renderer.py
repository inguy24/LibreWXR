# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
from PIL import Image

from librewxr.config import settings
from librewxr.sources.satellite.gmgsi.source import LAT_MAX as _GMGSI_LAT_MAX
from librewxr.sources.satellite.gmgsi.source import LAT_MIN as _GMGSI_LAT_MIN
from librewxr.tiles.coordinates import tile_pixel_latlons

# Smoothstep alpha attenuation across the last few degrees of GMGSI
# disk coverage so the ±72.7° horizontal cutoffs fade into the basemap
# instead of reading as a hard line at low zoom.  2° is ~220 km at the
# equator — visible as a soft fade without eating significant valid data.
_DISK_EDGE_FEATHER_DEGREES = 2.0


# GMGSI LW encoded value at the cloud threshold — pixels colder than
# this (encoded > threshold) are treated as cloud and rendered opaque;
# warmer pixels (ground / ocean / low cloud) ramp to fully transparent.
# Roughly 270 K on GMGSI's 0–255 brightness-temperature scale.
_LW_CLOUD_THRESHOLD = 25.0
_LW_CLOUD_MAX = 140.0


def _lw_brightness_and_alpha(
    encoded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute float32 brightness + alpha planes for one LW grid.

    Shared between the LW-only renderer and the composite renderer so
    the night side of the composite matches stand-alone LW exactly.
    Returns ``(brightness, alpha)`` both shaped like ``encoded`` with
    values in [0, 255] and [0, 1] respectively.
    """
    no_data = encoded == 0
    cloud_ramp = np.clip(
        (encoded.astype(np.float32) - _LW_CLOUD_THRESHOLD)
        / (_LW_CLOUD_MAX - _LW_CLOUD_THRESHOLD),
        0.0, 1.0,
    )
    alpha = np.power(cloud_ramp, 0.7)
    alpha = np.where(no_data, 0.0, alpha)
    brightness = encoded.astype(np.float32)
    return brightness, alpha


def _pack_rgba(brightness: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build an RGBA uint8 array with a slight cool tint in the blue."""
    rgba = np.zeros((*brightness.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(brightness + 5.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def _disk_edge_feather(lat_grid: np.ndarray) -> np.ndarray:
    """Smoothstep alpha multiplier that fades the disk's lat edges.

    Returns 1.0 anywhere more than ``_DISK_EDGE_FEATHER_DEGREES`` inside
    the GMGSI coverage band, ramping down to 0.0 at the edge via the
    cubic smoothstep ``3t² − 2t³`` (gentler than a linear ramp).  Pixels
    already outside the disk get 0 from the clip, which is harmless —
    their alpha was already 0 from the no-data sentinel.
    """
    edge_distance = np.minimum(
        _GMGSI_LAT_MAX - lat_grid,
        lat_grid - _GMGSI_LAT_MIN,
    )
    t = np.clip(edge_distance / _DISK_EDGE_FEATHER_DEGREES, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def render_gmgsi_tile(
    source,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a single-channel LW tile.

    Used as the fallback when VIS is unavailable.  Cold cloud tops are
    rendered as opaque bright pixels; warm ground / ocean / low cloud
    fades to transparent via the shared LW threshold ramp.  Same math
    as the LW half of the composite, served stand-alone.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    encoded = source.sample(lat_grid, lon_grid, timestamp)
    brightness, alpha = _lw_brightness_and_alpha(encoded)
    alpha = alpha * _disk_edge_feather(lat_grid)
    rgba = _pack_rgba(brightness, alpha)
    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def render_gmgsi_composite_tile(
    lw_source,
    vis_source,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a VIS-over-LW composite tile.

    The LW channel forms the base: cold cloud tops on a transparent
    map, same threshold ramp as the stand-alone LW renderer.  The VIS
    channel paints reflected sunlight on top with ``alpha = vis/255``,
    so the day side shows the natural view-from-space (clouds, land,
    ocean) and the night side falls through to LW IR.  The terminator
    crossfade emerges from the underlying VIS reflectance field — no
    sun-angle math required.

    Standard "VIS over LW" alpha composite::

        out_color = vis_color * vis_alpha + lw_color * (1 - vis_alpha)
        out_alpha = vis_alpha + lw_alpha * (1 - vis_alpha)

    Where VIS is zero (night side, or outside the disk), the formula
    collapses to LW alone, which is what we want.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    lw_encoded = lw_source.sample(lat_grid, lon_grid, timestamp)
    vis_encoded = vis_source.sample(lat_grid, lon_grid, timestamp)

    lw_brightness, lw_alpha = _lw_brightness_and_alpha(lw_encoded)

    # VIS uses its encoded value directly as both luminance and alpha.
    # The night side and outside-disk pixels are already 0 so they
    # contribute nothing to the composite without an explicit mask.
    vis_brightness = vis_encoded.astype(np.float32)
    vis_alpha = vis_encoded.astype(np.float32) / 255.0

    inv_vis_alpha = 1.0 - vis_alpha
    out_brightness = vis_brightness * vis_alpha + lw_brightness * inv_vis_alpha
    out_alpha = vis_alpha + lw_alpha * inv_vis_alpha
    out_alpha = out_alpha * _disk_edge_feather(lat_grid)

    rgba = _pack_rgba(out_brightness, out_alpha)
    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def render_geo_satellite_tile(
    ir_source,
    vis_source,    # can be None (IR-only mode)
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render an opaque satellite tile for geostationary sources (GOES, Himawari).

    Unlike the GMGSI renderers, this produces fully opaque tiles: pixels with
    data receive alpha=255 and pixels outside the satellite disk (no-data) receive
    alpha=0.  No disk-edge feathering is applied — the no-data mask already marks
    the GOES/Himawari disk boundary cleanly.  Imagery is grayscale (R=G=B), with
    no blue tint.

    When ``vis_source`` is provided, VIS is composited over IR using the same
    VIS-over-IR alpha math as the GMGSI composite renderer.  When ``vis_source``
    is ``None``, IR brightness is used directly.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)

    ir_encoded = ir_source.sample(lat_grid, lon_grid, timestamp)

    ir_brightness = ir_encoded.astype(np.float32)

    if vis_source is not None:
        vis_encoded = vis_source.sample(lat_grid, lon_grid, timestamp)
        vis_brightness = vis_encoded.astype(np.float32)
        vis_alpha = vis_encoded.astype(np.float32) / 255.0
        out_brightness = vis_brightness * vis_alpha + ir_brightness * (1.0 - vis_alpha)
        has_data = (ir_encoded > 0) | (vis_encoded > 0)
    else:
        out_brightness = ir_brightness
        has_data = ir_encoded > 0

    rgba = np.zeros((*out_brightness.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(out_brightness, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(out_brightness, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(out_brightness, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(has_data, np.uint8(255), np.uint8(0))

    return _encode_image(Image.fromarray(rgba, "RGBA"), fmt)


def render_multi_satellite_tile(
    sources: list[tuple[object, object | None, float]],
    z: int,
    x: int,
    y: int,
    timestamp: int,
    tile_size: int = 256,
    fmt: str = "png",
) -> bytes | None:
    """Render a satellite tile by selecting the best single source per tile.

    Each entry in *sources* is ``(ir_source, vis_source_or_None, sat_lon)``
    where ``sat_lon`` is the sub-satellite longitude in degrees.

    Selection: compute the tile center longitude, then pick the source
    whose sub-satellite point is closest — that satellite has the lowest
    zenith angle and therefore the best data for this tile.  If that
    source's scan grid doesn't cover the tile center (e.g. CONUS sector
    edge), fall through to the next-closest source.

    No per-pixel blending — each tile comes from exactly one satellite.
    This prevents artifacts where two satellites disagree on brightness
    at individual pixels due to different view angles.
    """
    # Compute tile center for source selection
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    center_lat = float(lat_grid[tile_size // 2, tile_size // 2])
    center_lon = float(lon_grid[tile_size // 2, tile_size // 2])

    # Rank sources by angular distance from tile center to sub-satellite point
    ranked = sorted(
        enumerate(sources),
        key=lambda item: min(
            abs(center_lon - item[1][2]),
            360.0 - abs(center_lon - item[1][2]),
        ),
    )

    # Pick the closest source whose scan grid covers the tile center
    chosen_ir = None
    chosen_vis = None
    for _idx, (ir_source, vis_source, _sat_lon) in ranked:
        if _source_covers_point(ir_source, center_lat, center_lon):
            chosen_ir = ir_source
            chosen_vis = vis_source
            break

    if chosen_ir is None:
        _, (chosen_ir, chosen_vis, _) = ranked[0]

    return render_geo_satellite_tile(
        ir_source=chosen_ir,
        vis_source=chosen_vis,
        z=z, x=x, y=y,
        tile_size=tile_size,
        timestamp=timestamp,
        fmt=fmt,
    )


def _source_covers_point(source, lat: float, lon: float) -> bool:
    """Check if a geostationary source's scan grid covers a given point."""
    from librewxr.tiles.geostationary import forward as geo_forward

    if source._x_vec is None or source._y_vec is None:
        return False

    x_ang, y_ang = geo_forward(
        np.array([lat], dtype=np.float64),
        np.array([lon], dtype=np.float64),
        source.sat_lon,
        source.sat_height,
    )

    if np.isnan(x_ang[0]) or np.isnan(y_ang[0]):
        return False

    x_step = (source._x_vec[-1] - source._x_vec[0]) / (source._grid_width - 1)
    y_step = (source._y_vec[0] - source._y_vec[-1]) / (source._grid_height - 1)

    col = (x_ang[0] - source._x_vec[0]) / x_step
    row = (source._y_vec[0] - y_ang[0]) / y_step

    return 0 <= col < source._grid_width and 0 <= row < source._grid_height


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    if fmt == "webp":
        q = settings.webp_quality
        if q >= 100:
            img.save(buf, format="WEBP", lossless=True)
        else:
            img.save(buf, format="WEBP", quality=q)
    else:
        img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
