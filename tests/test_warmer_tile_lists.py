# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Regression guard: the warmer's tile lists respect region overlap at ALL zooms.

Before 2026-08-08, zooms 0..warm_overview_zoom enumerated the full world
checkerboard (1+4+16+64+256 = 341 tiles at zooms 0-4) regardless of the
enabled regions, pre-rendering blank tiles for areas with no data and
permanently pinning a per-tile coordinate grid for each. These tests FAIL
against that code: it returns all 2**z x 2**z tiles at low zooms.
"""
import pytest

from librewxr.data.regions import REGIONS
from librewxr.tiles.coordinates import tile_overlaps_region
from librewxr.tiles.warmer import TileWarmer

pytestmark = pytest.mark.tiles

ENABLED = ["USCOMP"]


def _lists(max_zoom=4, max_zoom_regional=6):
    return TileWarmer._build_tile_lists(
        max_zoom, max_zoom_regional, max(max_zoom, max_zoom_regional), ENABLED
    )


class TestBuildTileLists:
    def test_low_zooms_exclude_non_overlapping_tiles(self):
        """The old code returned the full world checkerboard at z<=max_zoom."""
        tiles_by_zoom = _lists()
        region = REGIONS["USCOMP"]
        for z in (2, 3, 4):
            n = 2**z
            tiles = tiles_by_zoom[z]
            assert len(tiles) < n * n, (
                f"z={z}: expected fewer than the full {n * n}-tile world "
                f"checkerboard, got {len(tiles)}"
            )
            for x, y in tiles:
                assert tile_overlaps_region(region, z, x, y), (
                    f"z={z} tile ({x},{y}) does not overlap USCOMP"
                )

    def test_every_zoom_level_is_filtered_identically(self):
        """No zoom level may bypass the overlap filter."""
        tiles_by_zoom = _lists()
        region = REGIONS["USCOMP"]
        for z, tiles in tiles_by_zoom.items():
            for x, y in tiles:
                assert tile_overlaps_region(region, z, x, y)

    def test_overlapping_tiles_are_present(self):
        """The filter must not over-prune: known-overlapping tiles stay.

        The z=0 world tile always overlaps any region; at each deeper
        zoom at least one tile must cover the region's center.
        """
        tiles_by_zoom = _lists()
        assert (0, 0) in tiles_by_zoom[0]
        for z in range(1, 7):
            assert tiles_by_zoom[z], f"z={z} list is empty"
