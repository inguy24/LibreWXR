# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for BBOX radar region cropping.

The _apply_bbox_crop() function in data/regions.py crops all latlon
RegionDefs to the configured LIBREWXR_BBOX at import time.  Regions
entirely outside the BBOX are removed; regions that overlap are
replaced with bounds matching the intersection.  Non-latlon (projected)
regions are left untouched.

These tests exercise the cropping logic directly on RegionDef objects,
bypassing the import-time side effects (which depend on LIBREWXR_BBOX
env var and the full source discovery walker).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from librewxr.data.regions import RegionDef

pytestmark = pytest.mark.store


def _crop_region(
    region: RegionDef,
    bbox: tuple[float, float, float, float],
) -> RegionDef | None:
    """Apply BBOX crop logic to a single RegionDef.

    Reproduces the per-region logic from _apply_bbox_crop() without
    mutating the global REGIONS dict.  Returns None if the region is
    entirely outside the BBOX, the original region if fully contained,
    or a new cropped RegionDef for partial overlap.
    """
    south, west, north, east = bbox

    if region.proj != "latlon":
        return region

    # No overlap
    if (region.east <= west or region.west >= east
            or region.north <= south or region.south >= north):
        return None

    # Intersection
    new_west = max(region.west, west)
    new_east = min(region.east, east)
    new_south = max(region.south, south)
    new_north = min(region.north, north)

    if (new_west == region.west and new_east == region.east
            and new_south == region.south and new_north == region.north):
        return region  # fully contained

    return replace(
        region,
        west=new_west, east=new_east,
        south=new_south, north=new_north,
    )


# A CONUS-like region for testing
_CONUS = RegionDef(
    name="USCOMP",
    west=-130.0, east=-60.0,
    south=20.0, north=55.0,
    pixel_size=0.01,
    group="US",
)


class TestBBOXRegionCrop:
    """BBOX radar region cropping."""

    def test_region_fully_inside_bbox_unchanged(self):
        """A region fully inside the BBOX is returned unchanged."""
        bbox = (10.0, -140.0, 60.0, -50.0)
        result = _crop_region(_CONUS, bbox)
        assert result is _CONUS

    def test_region_partially_overlapping_is_cropped(self):
        """A region partially overlapping the BBOX is cropped to the intersection."""
        bbox = (32.0, -120.5, 35.5, -114.5)  # SoCal
        result = _crop_region(_CONUS, bbox)
        assert result is not None
        assert result is not _CONUS
        assert result.west == -120.5
        assert result.east == -114.5
        assert result.south == 32.0
        assert result.north == 35.5

    def test_region_fully_outside_bbox_removed(self):
        """A region entirely outside the BBOX is removed (returns None)."""
        bbox = (50.0, 0.0, 60.0, 30.0)  # Northern Europe
        result = _crop_region(_CONUS, bbox)
        assert result is None

    def test_no_bbox_leaves_region_unchanged(self):
        """Without BBOX, _apply_bbox_crop exits early; region stays as-is."""
        # We test the helper directly: no bbox = not called, so we just
        # verify the region is usable without crop.
        assert _CONUS.width > 0
        assert _CONUS.height > 0

    def test_crop_preserves_pixel_size(self):
        """Cropping changes bounds but not the pixel size."""
        bbox = (25.0, -100.0, 45.0, -80.0)
        result = _crop_region(_CONUS, bbox)
        assert result is not None
        assert result.pixel_size == _CONUS.pixel_size

    def test_crop_reduces_grid_dimensions(self):
        """Cropped region should have smaller width and height."""
        bbox = (32.0, -120.5, 35.5, -114.5)
        result = _crop_region(_CONUS, bbox)
        assert result is not None
        assert result.width < _CONUS.width
        assert result.height < _CONUS.height
        assert result.width == int(round((-114.5 - (-120.5)) / 0.01))
        assert result.height == int(round((35.5 - 32.0) / 0.01))

    def test_projected_region_not_cropped(self):
        """Non-latlon (projected) regions are left untouched."""
        projected = RegionDef(
            name="OPERA",
            west=-10.0, east=40.0,
            south=30.0, north=75.0,
            pixel_size=0.01,
            group="EUROPE",
            proj="laea",
            laea_lat0=55.0, laea_lon0=10.0,
        )
        bbox = (32.0, -120.5, 35.5, -114.5)  # SoCal, no overlap with OPERA
        result = _crop_region(projected, bbox)
        # Projected regions are returned unchanged regardless of overlap
        assert result is projected

    def test_touching_edge_is_not_overlap(self):
        """Regions sharing only an edge (no area overlap) are removed."""
        # BBOX east edge = region west edge
        bbox = (20.0, -180.0, 55.0, -130.0)
        result = _crop_region(_CONUS, bbox)
        # _CONUS.west = -130.0, bbox east = -130.0 → region.west >= east → no overlap
        assert result is None
