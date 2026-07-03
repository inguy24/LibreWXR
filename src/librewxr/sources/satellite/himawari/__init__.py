# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Himawari-9 AHI satellite source package.

High-resolution (2 km IR, 1 km VIS) geostationary imagery for
Asia-Pacific at 10-minute cadence from ``s3://noaa-himawari9/``.

When ``multi_satellite`` is True, enabled whenever the operator's BBOX
overlaps Himawari's geostationary disk (checked via forward projection,
not longitude thresholds — correctly handles Pacific BBOXes with
negative longitudes like Hawaii at -160°W).  When ``multi_satellite``
is False, center-longitude selection: 60°E to 180°E.

Returns ``[]`` for stations outside coverage, falling through to GMGSI
as the global fallback.
"""
from __future__ import annotations

from librewxr.sources._base import SatelliteContribution
from librewxr.sources.satellite._geo_base import bbox_overlaps_disk

from .source import HIMAWARI_HEIGHT, HIMAWARI_LON, HimawariIRSource, HimawariVISSource

__all__ = ["HimawariIRSource", "HimawariVISSource", "satellite_provider"]


def _center_longitude(settings) -> float | None:
    bbox = getattr(settings, "get_bbox", lambda: None)()
    if bbox is not None:
        _, west, _, east = bbox
        return (west + east) / 2.0

    station_lon = getattr(settings, "station_lon", None)
    if station_lon is not None:
        return float(station_lon)

    return None


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return Himawari IR + VIS contributions for Asia-Pacific stations.

    When ``multi_satellite`` is True, uses the geostationary forward
    projection to check whether any part of the BBOX is visible to
    Himawari-9 (at 140.7°E).  This correctly handles Pacific BBOXes
    with negative longitudes (e.g. Hawaii at -160°W, Guam at 144°E)
    that the previous simple longitude-range check missed.

    When ``multi_satellite`` is False, falls back to center-longitude
    selection: 60°E to 180°E.
    """
    if not getattr(settings, "himawari_enabled", True):
        return []

    center_lon = _center_longitude(settings)
    bbox = getattr(settings, "get_bbox", lambda: None)()

    disk_overlap = False
    if getattr(settings, "multi_satellite", True) and bbox is not None:
        disk_overlap = bbox_overlaps_disk(bbox, HIMAWARI_LON, HIMAWARI_HEIGHT)

    if disk_overlap:
        pass
    elif center_lon is None:
        return []
    elif not (60.0 <= center_lon <= 180.0):
        return []

    per_source = getattr(settings, "himawari_max_frames", 0)
    retention = per_source if per_source > 0 else getattr(settings, "satellite_max_frames", 36)
    contributions: list[SatelliteContribution] = []

    if getattr(settings, "himawari_ir_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=HimawariIRSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                ),
                priority=5,
                name="Himawari-9 IR",
                slug="himawari9_ir_grid",
            ),
        )

    if getattr(settings, "himawari_vis_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=HimawariVISSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                ),
                priority=6,
                name="Himawari-9 VIS",
                slug="himawari9_vis_grid",
            ),
        )

    return contributions
