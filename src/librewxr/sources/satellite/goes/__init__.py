# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""GOES-18 / GOES-19 ABI satellite source package.

High-resolution (2 km IR, 2 km VIS) geostationary imagery for the
Americas at 5-minute cadence.  When ``multi_satellite`` is True
(default), both GOES-18 and GOES-19 are enabled if the BBOX is visible
to each satellite's disk — the per-pixel compositor fills coverage gaps
from either satellite's CONUS sector scan limits.  When
``multi_satellite`` is False, auto-selects one satellite based on center
longitude.

When the operator's location is not in the Americas (or cannot be
determined), returns ``[]`` so GMGSI takes over as the global fallback.
"""
from __future__ import annotations

from librewxr.sources._base import SatelliteContribution
from librewxr.sources.satellite._geo_base import bbox_overlaps_disk

from .source import (
    GOES18IRSource,
    GOES18VISSource,
    GOES19IRSource,
    GOES19VISSource,
    GOES_HEIGHT,
)

__all__ = [
    "GOES18IRSource",
    "GOES18VISSource",
    "GOES19IRSource",
    "GOES19VISSource",
    "satellite_provider",
]

_GOES18_LON = -137.0
_GOES19_LON = -75.2


def _center_longitude(settings) -> float | None:
    """Determine the operator's center longitude from BBOX or station_lon."""
    bbox = getattr(settings, "get_bbox", lambda: None)()
    if bbox is not None:
        _, west, _, east = bbox
        return (west + east) / 2.0

    station_lon = getattr(settings, "station_lon", None)
    if station_lon is not None:
        return float(station_lon)

    return None


def _goes18_contributions(
    settings, cache_dir, retention, bbox, vis_downsample, cadence,
) -> list[SatelliteContribution]:
    """Build GOES-18 (West) IR + VIS contributions."""
    contribs: list[SatelliteContribution] = []
    if getattr(settings, "goes_ir_enabled", True):
        contribs.append(
            SatelliteContribution(
                instance=GOES18IRSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                    cadence_override=cadence,
                ),
                priority=5,
                name="GOES-18 IR",
                slug="goes18_ir_grid",
            ),
        )
    if getattr(settings, "goes_vis_enabled", True):
        contribs.append(
            SatelliteContribution(
                instance=GOES18VISSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                    downsample_factor=vis_downsample,
                    cadence_override=cadence,
                ),
                priority=6,
                name="GOES-18 VIS",
                slug="goes18_vis_grid",
            ),
        )
    return contribs


def _goes19_contributions(
    settings, cache_dir, retention, bbox, vis_downsample, cadence,
) -> list[SatelliteContribution]:
    """Build GOES-19 (East) IR + VIS contributions."""
    contribs: list[SatelliteContribution] = []
    if getattr(settings, "goes_ir_enabled", True):
        contribs.append(
            SatelliteContribution(
                instance=GOES19IRSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                    cadence_override=cadence,
                ),
                priority=5,
                name="GOES-19 IR",
                slug="goes19_ir_grid",
            ),
        )
    if getattr(settings, "goes_vis_enabled", True):
        contribs.append(
            SatelliteContribution(
                instance=GOES19VISSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                    downsample_factor=vis_downsample,
                    cadence_override=cadence,
                ),
                priority=6,
                name="GOES-19 VIS",
                slug="goes19_vis_grid",
            ),
        )
    return contribs


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return GOES IR + VIS contributions when the station is in the Americas.

    Selection logic:
    - Longitude between -170° and -30°: GOES coverage zone
    - When ``multi_satellite`` is True (default): enable each GOES
      satellite whose geostationary disk overlaps the BBOX.  Each
      satellite's CONUS sector has limited cross-hemisphere scan extent
      (the boundary is curved in lat/lon space because scan angles are
      fixed in geostationary projection), so using both ensures full
      BBOX coverage through the per-pixel compositor.
    - When ``multi_satellite`` is False: single satellite selected by
      center longitude (west of -100° → GOES-18, east → GOES-19).
    - Outside Americas: return [] (fall through to Himawari or GMGSI).
    """
    if not getattr(settings, "goes_enabled", True):
        return []

    center_lon = _center_longitude(settings)
    if center_lon is None:
        return []

    if not (-170.0 <= center_lon <= -30.0):
        return []

    per_source = getattr(settings, "goes_max_frames", 0)
    retention = per_source if per_source > 0 else getattr(settings, "satellite_max_frames", 36)
    bbox = getattr(settings, "get_bbox", lambda: None)()
    vis_hires = getattr(settings, "goes_vis_hires", False)
    vis_downsample = 1 if vis_hires else 4  # 0.5 km native -> 2 km default
    cadence = getattr(settings, "satellite_cadence", 0)

    contributions: list[SatelliteContribution] = []

    if getattr(settings, "multi_satellite", True) and bbox is not None:
        if bbox_overlaps_disk(bbox, _GOES18_LON, GOES_HEIGHT):
            contributions.extend(
                _goes18_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
            )
        if bbox_overlaps_disk(bbox, _GOES19_LON, GOES_HEIGHT):
            contributions.extend(
                _goes19_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
            )
        if not contributions:
            # BBOX visible from neither disk — shouldn't happen for Americas,
            # but fall through to center-based selection as safety net.
            if center_lon < -100.0:
                contributions.extend(
                    _goes18_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
                )
            else:
                contributions.extend(
                    _goes19_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
                )
    elif center_lon < -100.0:
        contributions.extend(
            _goes18_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
        )
    else:
        contributions.extend(
            _goes19_contributions(settings, cache_dir, retention, bbox, vis_downsample, cadence),
        )

    return contributions
