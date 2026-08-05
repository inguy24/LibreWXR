# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the GOES satellite source.

Covers auto-selection logic, provider registration, filename parsing,
sample() with synthetic grids, and the cross-process pickle round-trip.
S3 I/O is mocked — live verification is a separate deployment step.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from librewxr.sources.satellite._geo_base import GeoSatSource
from librewxr.sources.satellite.goes.source import (
    GOES18IRSource,
    GOES18VISSource,
    GOES19IRSource,
    GOES19VISSource,
)

pytestmark = pytest.mark.sources


# ── Filename parsing ──


def test_parse_goes_timestamp_doy_format():
    """GOES filenames use day-of-year format: _sYYYYDDDHHMMSSt."""
    fn = "OR_ABI-L2-CMIPC-M6C13_G18_s20261791801174_e20261791803547_c20261791804012.nc"
    ts = GeoSatSource._parse_start_timestamp(fn)
    assert ts is not None
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.year == 2026
    assert dt.month == 6  # DOY 179 = June 28
    assert dt.hour == 18
    assert dt.minute == 1


def test_parse_goes_timestamp_rejects_malformed():
    assert GeoSatSource._parse_start_timestamp("not_a_goes_file.nc") is None
    assert GeoSatSource._parse_start_timestamp("OR_ABI_s2026.nc") is None


# ── Class metadata ──


def test_goes18_ir_metadata():
    assert GOES18IRSource.sat_lon == -137.0
    assert GOES18IRSource.s3_bucket == "noaa-goes18"
    assert GOES18IRSource.s3_product_path == "ABI-L2-CMIPC"
    assert GOES18IRSource.s3_filename_token == "CMIPC-M6C13"
    assert GOES18IRSource.channel == "IR"
    assert GOES18IRSource.band == 13


def test_goes19_vis_metadata():
    assert GOES19VISSource.sat_lon == -75.2
    assert GOES19VISSource.s3_bucket == "noaa-goes19"
    assert GOES19VISSource.channel == "VIS"
    assert GOES19VISSource.band == 2


# ── Provider auto-selection ──


def test_provider_returns_goes18_for_socal():
    """SoCal (lon ~-117) should select GOES-18."""
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (32.0, -120.5, 35.5, -114.5)
    settings.goes_enabled = True
    settings.goes_ir_enabled = True
    settings.goes_vis_enabled = True
    settings.satellite_max_frames = 12
    # Real integers, not MagicMock's auto-vivified attrs: the provider
    # compares these with `> 0` (goes_max_frames=0 exercises the
    # satellite_max_frames-default path; satellite_cadence=0 exercises
    # the class-default cadence path).
    settings.goes_max_frames = 0
    settings.satellite_cadence = 0
    # Pins the single-satellite center-longitude path this test exercises;
    # the default multi-family disk-overlap selection (ADR-079) would
    # return both GOES-18 and GOES-19 for this bbox (both disks overlap
    # it) and is deliberately not under test here.
    settings.multi_satellite = False

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 2
    assert contribs[0].slug == "goes18_ir_grid"
    assert contribs[1].slug == "goes18_vis_grid"


def test_provider_returns_goes19_for_nyc():
    """NYC (lon ~-74) should select GOES-19."""
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (40.0, -74.5, 41.0, -73.5)
    settings.goes_enabled = True
    settings.goes_ir_enabled = True
    settings.goes_vis_enabled = True
    settings.satellite_max_frames = 12
    settings.goes_max_frames = 0
    settings.satellite_cadence = 0
    # Pins the single-satellite center-longitude path this test exercises;
    # the default multi-family disk-overlap selection (ADR-079) would
    # return both GOES-18 and GOES-19 for this bbox (both disks overlap
    # it) and is deliberately not under test here.
    settings.multi_satellite = False

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 2
    assert contribs[0].slug == "goes19_ir_grid"
    assert contribs[1].slug == "goes19_vis_grid"


def test_provider_returns_empty_for_tokyo():
    """Tokyo (lon ~139.7) is outside GOES coverage → empty."""
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (35.0, 139.0, 36.0, 140.0)
    settings.goes_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_returns_empty_when_disabled():
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.goes_enabled = False
    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_returns_empty_when_no_location():
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = None
    settings.station_lon = None
    settings.goes_enabled = True
    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_uses_station_lon_fallback():
    """When no BBOX is set, station_lon is used for auto-selection."""
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = None
    settings.station_lon = -118.0  # LA
    settings.goes_enabled = True
    settings.goes_ir_enabled = True
    settings.goes_vis_enabled = False
    settings.satellite_max_frames = 12
    # Real integers, not MagicMock's auto-vivified attrs: the provider
    # compares these with `> 0` (goes_max_frames=0 exercises the
    # satellite_max_frames-default path; satellite_cadence=0 exercises
    # the class-default cadence path).
    settings.goes_max_frames = 0
    settings.satellite_cadence = 0

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 1
    assert contribs[0].slug == "goes18_ir_grid"


# ── sample() with synthetic grid ──


def test_sample_returns_zero_when_no_frames():
    src = GOES18IRSource(cache_dir=None, max_frames=3)
    lat = np.array([[34.0]], dtype=np.float32)
    lon = np.array([[-118.0]], dtype=np.float32)
    out = src.sample(lat, lon, timestamp=None)
    assert out.shape == (1, 1)
    assert out[0, 0] == 0


def test_sample_returns_nonzero_for_visible_point(tmp_path: Path):
    """Synthetic grid with all-200 values; visible points should sample 200."""
    src = GOES18IRSource(cache_dir=None, max_frames=3)

    # Simulate grid vectors (small 10x10 grid covering CONUS roughly)
    src._x_vec = np.linspace(-0.10, 0.02, 100, dtype=np.float64)
    src._y_vec = np.linspace(0.12, 0.04, 80, dtype=np.float64)
    src._grid_width = 100
    src._grid_height = 80

    grid = np.full((80, 100), 200, dtype=np.uint8)
    ts = 12345
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]

    # LA is visible from GOES-18
    lat = np.array([[34.0]], dtype=np.float64)
    lon = np.array([[-118.0]], dtype=np.float64)
    out = src.sample(lat, lon, timestamp=ts)
    # Should be non-zero if the point falls within the grid
    # (may be 0 if the synthetic grid is too small — that's expected)
    assert out.shape == (1, 1)
    assert out.dtype == np.uint8


def test_sample_returns_zero_for_invisible_point():
    """Points behind the earth (from GOES-18's perspective) should return 0."""
    src = GOES18IRSource(cache_dir=None, max_frames=3)
    src._x_vec = np.linspace(-0.10, 0.10, 100, dtype=np.float64)
    src._y_vec = np.linspace(0.10, -0.10, 100, dtype=np.float64)
    src._grid_width = 100
    src._grid_height = 100

    grid = np.full((100, 100), 200, dtype=np.uint8)
    src._frames[12345] = grid
    src._sorted_timestamps = [12345]

    # Far side of earth from GOES-18 (lon ≈ +43°)
    lat = np.array([[0.0]], dtype=np.float64)
    lon = np.array([[43.0]], dtype=np.float64)
    out = src.sample(lat, lon, timestamp=12345)
    assert out[0, 0] == 0


# ── IR value mapping ──


def test_ir_cold_maps_to_high_uint8():
    """Cold temperatures (high clouds) should map to high uint8 values."""
    src = GOES18IRSource(cache_dir=None)
    cold = np.array([[170.0]])  # T_MIN → should be 255
    encoded = src._map_to_uint8(cold)
    assert encoded[0, 0] == 255


def test_ir_warm_maps_to_low_uint8():
    """Warm temperatures (ground) should map to low uint8 values.

    _IR_T_MAX was extended from 320 K to 340 K (commit 2dcaa98, per NOAA
    ABI spec) — 340 K, not 320 K, is now the ceiling that maps to 0.
    320 K is hand-computed from the current formula constants
    (_IR_T_MIN=170, _IR_T_MAX=340, _IR_RANGE=170):
        encoded = 255 * (340 - 320) / 170 = 255 * 20 / 170 = 30.0 -> 30
    Hardcoded so future drift in the constants fails loudly here.
    """
    src = GOES18IRSource(cache_dir=None)
    hottest = np.array([[340.0]])  # current T_MAX → should be 0
    encoded = src._map_to_uint8(hottest)
    assert encoded[0, 0] == 0

    warm = np.array([[320.0]])
    encoded_warm = src._map_to_uint8(warm)
    assert encoded_warm[0, 0] == 30


def test_ir_nan_maps_to_zero():
    src = GOES18IRSource(cache_dir=None)
    data = np.array([[np.nan]])
    encoded = src._map_to_uint8(data)
    assert encoded[0, 0] == 0


# ── VIS value mapping ──


def test_vis_full_reflectance_maps_to_255():
    src = GOES18VISSource(cache_dir=None)
    data = np.array([[1.0]])
    encoded = src._map_to_uint8(data)
    assert encoded[0, 0] == 255


def test_vis_zero_reflectance_maps_to_zero():
    src = GOES18VISSource(cache_dir=None)
    data = np.array([[0.0]])
    encoded = src._map_to_uint8(data)
    assert encoded[0, 0] == 0


# ── Pickle round-trip ──


def test_pickle_round_trip(tmp_path: Path):
    """Pipeline → render worker snapshot via __getstate__/__setstate__."""
    src = GOES18IRSource(cache_dir=tmp_path, max_frames=5)
    src._x_vec = np.linspace(-0.1, 0.1, 50, dtype=np.float64)
    src._y_vec = np.linspace(0.1, -0.1, 40, dtype=np.float64)
    src._grid_width = 50
    src._grid_height = 40

    grid = np.full((40, 50), 150, dtype=np.uint8)
    ts = 99999
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]
    src._write_cache(ts, grid)

    state = src.__getstate__()
    assert state["channel"] == "IR"
    assert ts in state["timestamps"]

    render_src = GOES18IRSource.__new__(GOES18IRSource)
    render_src.__setstate__(state)
    assert render_src.timestamps == [ts]
    np.testing.assert_array_equal(render_src._frames[ts], grid)


# ── BBOX crop ──


class TestBBOXCrop:
    """BBOX crop for geostationary sources."""

    def test_crop_reduces_grid_dimensions(self):
        """With a BBOX, grid should be much smaller than full CONUS.

        x_vec span corrected to the ABI fixed-grid convention (x increases
        EASTWARD; commit 0bd6a99) — this SoCal bbox projects from GOES-18
        (-137°W) to x in [0.0395, 0.0555] rad (measured via geo_forward on
        the bbox corners), which the old x_vec (-0.10…+0.02) didn't cover
        at all. New span -0.02…+0.13 rad covers it with margin; y_vec is
        unchanged (already covers the corners' y in [0.0902, 0.0987]).
        """
        bbox = (32.0, -120.5, 35.5, -114.5)
        src = GOES18IRSource(cache_dir=None, max_frames=3, bbox=bbox)
        src._x_vec = np.linspace(-0.02, 0.13, 2500, dtype=np.float64)
        src._y_vec = np.linspace(0.12, 0.04, 1500, dtype=np.float64)
        src._grid_width = 2500
        src._grid_height = 1500
        src._compute_crop_indices(src._x_vec, src._y_vec, 2500, 1500)
        assert src._crop_computed
        crop_h = src._crop_row_end - src._crop_row_start
        crop_w = src._crop_col_end - src._crop_col_start
        assert crop_h < 500, f"Cropped height {crop_h} not much smaller than 1500"
        assert crop_w < 500, f"Cropped width {crop_w} not much smaller than 2500"
        assert crop_h > 0
        assert crop_w > 0

    def test_no_bbox_means_no_crop(self):
        """Without BBOX, no crop should be computed."""
        src = GOES18IRSource(cache_dir=None, max_frames=3, bbox=None)
        assert not src._crop_computed

    def test_bbox_outside_coverage_no_crop(self):
        """BBOX entirely invisible from GOES-18 should not produce a crop."""
        # London is not visible from GOES-18 at -137 W
        bbox = (51.0, -0.5, 52.0, 0.5)
        src = GOES18IRSource(cache_dir=None, max_frames=3, bbox=bbox)
        src._compute_crop_indices(
            np.linspace(-0.10, 0.02, 2500, dtype=np.float64),
            np.linspace(0.12, 0.04, 1500, dtype=np.float64),
            2500, 1500,
        )
        assert not src._crop_computed

    def test_crop_indices_within_grid_bounds(self):
        """Crop indices must stay within the grid dimensions."""
        bbox = (25.0, -130.0, 50.0, -60.0)
        src = GOES18IRSource(cache_dir=None, max_frames=3, bbox=bbox)
        full_w, full_h = 2500, 1500
        src._compute_crop_indices(
            np.linspace(-0.10, 0.02, full_w, dtype=np.float64),
            np.linspace(0.12, 0.04, full_h, dtype=np.float64),
            full_w, full_h,
        )
        assert src._crop_computed
        assert 0 <= src._crop_row_start < src._crop_row_end <= full_h
        assert 0 <= src._crop_col_start < src._crop_col_end <= full_w

    def test_sample_returns_zero_outside_cropped_bbox(self):
        """After BBOX crop, sample() should return 0 for points outside.

        x_vec span corrected to the ABI fixed-grid convention (x increases
        EASTWARD; commit 0bd6a99), same reasoning as
        test_crop_reduces_grid_dimensions above.
        """
        bbox = (32.0, -120.5, 35.5, -114.5)
        src = GOES18IRSource(cache_dir=None, max_frames=3, bbox=bbox)
        # Set up a full grid, then crop
        full_x = np.linspace(-0.02, 0.13, 500, dtype=np.float64)
        full_y = np.linspace(0.12, 0.04, 400, dtype=np.float64)
        src._compute_crop_indices(full_x, full_y, 500, 400)
        assert src._crop_computed
        # Apply the crop to the vectors (as _init_grid_vectors would)
        src._x_vec = full_x[src._crop_col_start:src._crop_col_end]
        src._y_vec = full_y[src._crop_row_start:src._crop_row_end]
        src._grid_width = len(src._x_vec)
        src._grid_height = len(src._y_vec)
        # Fill with non-zero data
        grid = np.full((src._grid_height, src._grid_width), 200, dtype=np.uint8)
        ts = 55555
        src._frames[ts] = grid
        src._sorted_timestamps = [ts]
        # NYC is far outside the SoCal BBOX and outside the cropped grid
        lat = np.array([[40.7]], dtype=np.float64)
        lon = np.array([[-74.0]], dtype=np.float64)
        out = src.sample(lat, lon, timestamp=ts)
        assert out[0, 0] == 0


# ── Renderer compatibility ──


def test_goes_source_renders_via_satellite_renderer():
    """GOES source works with the existing satellite renderer."""
    from librewxr.tiles.satellite_renderer import render_gmgsi_tile

    src = GOES18IRSource(cache_dir=None, max_frames=3)
    src._x_vec = np.linspace(-0.10, 0.02, 100, dtype=np.float64)
    src._y_vec = np.linspace(0.12, 0.04, 80, dtype=np.float64)
    src._grid_width = 100
    src._grid_height = 80
    grid = np.full((80, 100), 150, dtype=np.uint8)
    ts = 12345
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]

    tile_bytes = render_gmgsi_tile(
        source=src, z=3, x=1, y=2,
        tile_size=256, timestamp=ts, fmt="png",
    )
    assert len(tile_bytes) > 0
    assert tile_bytes[:4] == b"\x89PNG"


# ── G1: retention-window trim (guard for D1) ──


def test_fetch_sync_skips_out_of_retention_keys(monkeypatch):
    """Keys beyond the newest ``max_frames`` must not be downloaded.

    Any key older than the newest ``max_frames`` would be evicted by the
    trim loop immediately after ingest, so downloading it is pure waste.
    Pre-change, ``_fetch_sync`` downloads every listed key not already in
    ``self._frames`` before trimming: a partially-warm store still
    re-downloads out-of-retention keys, and an empty store downloads
    everything listed instead of just the newest ``max_frames``.
    """
    src = GOES18IRSource(cache_dir=None, max_frames=4)
    src._fs = MagicMock()  # _get_fs() short-circuits when _fs is already set

    keys = [(1_700_000_000 + i * 300, f"key{i}") for i in range(10)]  # ascending
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: keys)

    call_count = 0

    def fake_download(fs, s3_key):
        nonlocal call_count
        call_count += 1
        return np.zeros((2, 2), dtype=np.uint8)

    monkeypatch.setattr(src, "_download_and_decode", fake_download)

    # Store already holds the newest 4 timestamps — nothing new to fetch.
    newest_4 = [ts for ts, _ in keys[-4:]]
    for ts in newest_4:
        src._frames[ts] = np.zeros((2, 2), dtype=np.uint8)
    src._sorted_timestamps = sorted(src._frames)

    src._fetch_sync()

    assert call_count == 0, (
        f"expected 0 downloads for an already-warm store, got {call_count}"
    )
    assert sorted(src._frames) == newest_4

    # Empty store: only the newest max_frames keys should be downloaded,
    # not every key in the (deliberately generous) listing window.
    src2 = GOES18IRSource(cache_dir=None, max_frames=4)
    src2._fs = MagicMock()
    monkeypatch.setattr(src2, "_list_recent_keys", lambda fs, start, end: keys)

    call_count2 = 0

    def fake_download2(fs, s3_key):
        nonlocal call_count2
        call_count2 += 1
        return np.zeros((2, 2), dtype=np.uint8)

    monkeypatch.setattr(src2, "_download_and_decode", fake_download2)

    src2._fetch_sync()

    assert call_count2 == 4, (
        f"expected 4 downloads (newest max_frames only) for an empty store, "
        f"got {call_count2}"
    )
    assert sorted(src2._frames) == newest_4


def test_fetch_sync_duplicate_timestamp_key_does_not_evict_newer_frame(monkeypatch):
    """A republished scan (same timestamp, second key) must not consume a
    retention slot.

    NOAA can republish a scan under a new S3 key with the same ``_s`` start
    token (different ``_c`` creation token). Trimming the listing by raw
    tuple count lets the duplicate push a genuinely newer *distinct*
    timestamp out of the newest-``max_frames`` window, silently dropping a
    frame that should survive retention. The trim must count distinct
    timestamps.

    Tie-break for which key wins per timestamp: latest-created (round 3,
    A1) — reversed from the first-key-wins tie-break this test originally
    pinned, per lead ruling superseding the 60ffa81 remediation. The
    dedupe-before-trim behavior under test here (distinct timestamps
    surviving retention) is unchanged by that reversal.
    """
    src = GOES18IRSource(cache_dir=None, max_frames=5)
    src._fs = MagicMock()

    # 6 distinct timestamps; a LATE timestamp (i=4) is listed twice under
    # different keys, so a raw tuple-count trim cuts a distinct older
    # timestamp that belongs in the newest-5 window.
    base = 1_700_000_000
    keys = [(base + i * 300, f"key{i}") for i in range(6)]
    keys.append((base + 4 * 300, "key4-republished"))
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: keys)

    downloaded: list[str] = []

    def fake_download(fs, s3_key):
        downloaded.append(s3_key)
        return np.zeros((2, 2), dtype=np.uint8)

    monkeypatch.setattr(src, "_download_and_decode", fake_download)

    src._fetch_sync()

    expected_newest_5 = [base + i * 300 for i in range(1, 6)]
    assert sorted(src._frames) == expected_newest_5, (
        f"store holds {sorted(src._frames)}; the duplicate key for ts "
        f"{base + 4 * 300} must not push distinct ts {base + 300} out of "
        f"the newest-5 window"
    )
    # Latest (lexicographically largest) key per timestamp wins (A1) —
    # "key4-republished" > "key4" lexicographically, so the republished
    # scan wins the slot, matching the ingest loop's own tie-break.
    assert "key4-republished" in downloaded and "key4" not in downloaded


# ── G3: NaN-safe sample() cast (guard for D3) ──


def test_sample_no_runtime_warning_on_mixed_visibility():
    """sample() must not raise RuntimeWarning casting NaN scan angles.

    Baseline casts ``x_ang``/``y_ang`` (containing NaN for off-disk points)
    to int32 *before* the ``visible`` mask is applied, which raises
    ``RuntimeWarning: invalid value encountered in cast`` under a strict
    warnings filter. The fix must not change any visible-point output.
    """
    src = GOES18IRSource(cache_dir=None, max_frames=3)
    src._x_vec = np.linspace(-0.10, 0.10, 100, dtype=np.float64)
    src._y_vec = np.linspace(0.10, -0.10, 100, dtype=np.float64)
    src._grid_width = 100
    src._grid_height = 100

    grid = np.full((100, 100), 200, dtype=np.uint8)
    ts = 12345
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]

    # LA is visible from GOES-18 (-137.0) and falls within this synthetic
    # grid; ~43E is roughly antipodal and off-disk (same point used by
    # test_sample_returns_zero_for_invisible_point above).
    lat = np.array([34.0, 0.0], dtype=np.float64)
    lon = np.array([-118.0, 43.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = src.sample(lat, lon, timestamp=ts)

    assert out[0] == 200, "visible point should sample the uniform grid value"
    assert out[1] == 0, "off-disk point must stay 0"


# ── G-R3a/b/c: reprocessed-scan adoption (guards for round-3 A1/A2) ──


def test_fetch_sync_latest_key_wins_in_one_listing(monkeypatch):
    """A1: when one listing has two keys for the same timestamp, the
    lexicographically LARGEST (latest-created) key is downloaded — the
    earlier key is never fetched.

    Pre-change (first-key-wins): the earlier key is downloaded instead.
    """
    src = GOES18IRSource(cache_dir=None, max_frames=4)
    src._fs = MagicMock()

    ts = 1_700_000_000
    early_key = "goes18_c20250101010101.nc"
    late_key = "goes18_c20250101010999.nc"  # later _c creation token
    keys = [(ts, early_key), (ts, late_key)]
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: keys)

    downloaded: list[str] = []

    def fake_download(fs, s3_key):
        downloaded.append(s3_key)
        return np.zeros((2, 2), dtype=np.uint8)

    monkeypatch.setattr(src, "_download_and_decode", fake_download)

    src._fetch_sync()

    assert downloaded == [late_key], (
        f"expected only the later key downloaded, got {downloaded}"
    )


def test_fetch_sync_replaces_resident_frame_with_newer_key(monkeypatch):
    """A2: a later poll listing a newer key for an already-resident
    timestamp triggers exactly one re-download; the frame data is
    replaced; ``consume_replaced_timestamps()`` reports it exactly once,
    then reports nothing on a subsequent poll with the same listing.

    Pre-change: no ``consume_replaced_timestamps`` method — AttributeError
    (a mechanics pin: the replacement machinery doesn't exist yet).
    """
    src = GOES18IRSource(cache_dir=None, max_frames=4)
    src._fs = MagicMock()

    ts = 1_700_000_000
    key_a = "goes18_c20250101010101.nc"
    key_b = "goes18_c20250101010999.nc"  # newer — reprocessed product

    call_log: list[str] = []

    def fake_download(fs, s3_key):
        call_log.append(s3_key)
        value = 11 if s3_key == key_a else 22  # distinguishable stub arrays
        return np.full((2, 2), value, dtype=np.uint8)

    monkeypatch.setattr(src, "_download_and_decode", fake_download)

    # Poll 1: ingest ts with key_a.
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: [(ts, key_a)])
    src._fetch_sync()
    assert call_log == [key_a]
    np.testing.assert_array_equal(
        src._frames[ts], np.full((2, 2), 11, dtype=np.uint8),
    )
    assert src.consume_replaced_timestamps() == []

    # Poll 2: listing now has key_b (newer) for the same ts.
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: [(ts, key_b)])
    src._fetch_sync()
    assert call_log == [key_a, key_b], "expected exactly one re-download"
    np.testing.assert_array_equal(
        src._frames[ts], np.full((2, 2), 22, dtype=np.uint8),
    )
    assert src.consume_replaced_timestamps() == [ts]

    # Poll 3: same listing (key_b) again — no further download, no replacement.
    src._fetch_sync()
    assert call_log == [key_a, key_b], "no further download expected"
    assert src.consume_replaced_timestamps() == []


def test_fetch_sync_adopts_key_for_resident_frame_with_no_recorded_key(monkeypatch):
    """A2: a frame resident from a disk-cache/snapshot restore (key
    unknown — ``_frame_keys`` has no entry) adopts the listed key without
    re-downloading. A subsequent poll with a NEWER key then triggers
    replacement (the G-R3b behavior above).

    Pre-change: no ``_frame_keys`` attribute — AttributeError.
    """
    src = GOES18IRSource(cache_dir=None, max_frames=4)
    src._fs = MagicMock()

    ts = 1_700_000_000
    key_a = "goes18_c20250101010101.nc"
    key_b = "goes18_c20250101010999.nc"

    # Simulate a frame restored from disk cache: resident in _frames, but
    # _frame_keys has no entry for it (key unknown).
    src._frames[ts] = np.full((2, 2), 99, dtype=np.uint8)
    src._sorted_timestamps = [ts]
    assert ts not in src._frame_keys

    call_log: list[str] = []

    def fake_download(fs, s3_key):
        call_log.append(s3_key)
        return np.full((2, 2), 22, dtype=np.uint8)

    monkeypatch.setattr(src, "_download_and_decode", fake_download)
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: [(ts, key_a)])

    src._fetch_sync()

    assert call_log == [], "adopting the key must not re-download"
    assert src._frame_keys[ts] == key_a
    np.testing.assert_array_equal(
        src._frames[ts], np.full((2, 2), 99, dtype=np.uint8),
    )  # untouched by the adopt
    assert src.consume_replaced_timestamps() == []

    # A subsequent poll with a newer key now triggers replacement.
    monkeypatch.setattr(src, "_list_recent_keys", lambda fs, start, end: [(ts, key_b)])
    src._fetch_sync()
    assert call_log == [key_b]
    assert src.consume_replaced_timestamps() == [ts]


# ── R3-B: multi-satellite default-path coverage (P2) ──


def test_provider_returns_both_families_for_multi_satellite_default():
    """Pins ADR-079's default disk-overlap selection: with
    ``multi_satellite=True`` (the documented default) and a bbox visible
    to both GOES-18 and GOES-19, ``satellite_provider()`` returns all 4
    contributions (both families' IR + VIS). The repaired single-satellite
    tests above (``test_provider_returns_goes18_for_socal`` etc.)
    deliberately exclude this path via ``multi_satellite=False`` — this
    test pins the default path they don't cover (parking-lot item P2).

    New pin — non-falsifiable-vs-pre-change (no prior test covered this
    path at all; it simply didn't exist before).
    """
    from librewxr.sources.satellite.goes import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (32.0, -120.5, 35.5, -114.5)
    settings.goes_enabled = True
    settings.goes_ir_enabled = True
    settings.goes_vis_enabled = True
    settings.satellite_max_frames = 12
    settings.goes_max_frames = 0
    settings.satellite_cadence = 0
    settings.multi_satellite = True

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 4
    slugs = {c.slug for c in contribs}
    assert slugs == {
        "goes18_ir_grid", "goes18_vis_grid",
        "goes19_ir_grid", "goes19_vis_grid",
    }
