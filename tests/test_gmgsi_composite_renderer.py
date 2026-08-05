# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for the GMGSI tile renderers (LW + composite).

The composite is "VIS over LW" via standard alpha compositing.  These
tests use a duck-typed fake source returning a constant grid so we can
predict the exact post-blend pixel values without round-tripping S3 or
NetCDF decoding.

Tile coordinates ``z=2, x=2, y=1`` are chosen so the tile sits entirely
between lat 0° and +66.5° — well inside GMGSI's ±72.7° disk, where the
edge-feather alpha multiplier is 1.0 and doesn't perturb the predicted
output values.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from librewxr.tiles.satellite_renderer import (
    _lw_brightness_and_alpha,
    render_gmgsi_composite_tile,
    render_gmgsi_tile,
)

pytestmark = pytest.mark.tiles


class _ConstantSource:
    """Duck-typed GMGSI source returning a uniform encoded grid."""

    def __init__(self, value: int) -> None:
        self.value = value

    def sample(self, lat: np.ndarray, lon: np.ndarray, timestamp=None) -> np.ndarray:
        return np.full(lat.shape, self.value, dtype=np.uint8)


def _decode_png(png_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png_bytes)))


# ── LW alpha curve ──


def test_lw_helper_zero_encoded_is_transparent():
    encoded = np.array([0, 0, 0], dtype=np.uint8)
    _, alpha = _lw_brightness_and_alpha(encoded)
    assert np.all(alpha == 0.0)


def test_lw_helper_below_threshold_is_transparent():
    """Values at/below the current cloud threshold (25, not the old 110
    pre-38fcbfe/80a8bf1) are fully transparent; one value just above it
    is nonzero.

    Current constants: _LW_CLOUD_THRESHOLD=25.0, _LW_CLOUD_MAX=140.0.
    For 30 (just above threshold):
        ramp = (30 - 25) / (140 - 25) = 5 / 115 ≈ 0.043478
        alpha = ramp ** 0.7 ≈ 0.11138
    """
    encoded = np.array([5, 15, 25], dtype=np.uint8)
    _, alpha = _lw_brightness_and_alpha(encoded)
    assert np.all(alpha == 0.0)

    above = np.array([30], dtype=np.uint8)
    _, alpha_above = _lw_brightness_and_alpha(above)
    expected = ((30.0 - 25.0) / (140.0 - 25.0)) ** 0.7
    assert alpha_above[0] == pytest.approx(expected)
    assert alpha_above[0] > 0.0


def test_lw_helper_at_max_is_fully_opaque():
    encoded = np.array([255], dtype=np.uint8)
    _, alpha = _lw_brightness_and_alpha(encoded)
    assert alpha[0] == pytest.approx(1.0)


def test_lw_helper_brightness_passes_through():
    encoded = np.array([50, 180, 255], dtype=np.uint8)
    brightness, _ = _lw_brightness_and_alpha(encoded)
    assert np.array_equal(brightness, encoded.astype(np.float32))


# ── Composite renderer ──


def test_composite_pure_night_renders_lw_only():
    """VIS=0 collapses the composite to the LW threshold result."""
    lw = _ConstantSource(180)  # cold cloud
    vis = _ConstantSource(0)
    png = render_gmgsi_composite_tile(lw, vis, z=2, x=2, y=1, tile_size=32)
    composite = _decode_png(png)

    lw_only_png = render_gmgsi_tile(lw, z=2, x=2, y=1, tile_size=32)
    lw_only = _decode_png(lw_only_png)

    # Identical: VIS contributed nothing.
    assert np.array_equal(composite, lw_only)


def test_composite_pure_day_dominated_by_vis():
    """VIS=255 fully overrides LW (vis_alpha=1.0 → out=vis)."""
    lw = _ConstantSource(180)
    vis = _ConstantSource(255)
    png = render_gmgsi_composite_tile(lw, vis, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)

    # VIS brightness wins entirely; alpha is also 1.0.
    assert rgba[0, 0, 0] == 255
    assert rgba[0, 0, 1] == 255
    assert rgba[0, 0, 2] == 255  # clipped from 260 after cool tint
    assert rgba[0, 0, 3] == 255


def test_composite_twilight_blends_linearly():
    """VIS=128 → roughly 50/50 mix of VIS and LW brightness in the output."""
    lw_value = 180
    vis_value = 128
    lw = _ConstantSource(lw_value)
    vis = _ConstantSource(vis_value)

    png = render_gmgsi_composite_tile(lw, vis, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)

    vis_alpha = vis_value / 255.0
    expected = vis_value * vis_alpha + lw_value * (1.0 - vis_alpha)
    # Allow ±2 for PNG quantization + uint8 round-trip.
    assert abs(int(rgba[0, 0, 0]) - round(expected)) <= 2


def test_composite_outside_disk_is_fully_transparent():
    """Both channels reporting encoded=0 means nothing to render."""
    lw = _ConstantSource(0)
    vis = _ConstantSource(0)
    png = render_gmgsi_composite_tile(lw, vis, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)
    assert np.all(rgba[..., 3] == 0)


def test_composite_alpha_includes_lw_contribution_on_night_side():
    """Cold LW under VIS=0 stays visible (alpha > 0) — clouds on night side."""
    lw = _ConstantSource(220)  # very cold cloud
    vis = _ConstantSource(0)
    png = render_gmgsi_composite_tile(lw, vis, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)
    assert rgba[0, 0, 3] > 200  # well above transparent


# ── LW-only renderer ──


def test_lw_only_renderer_returns_valid_png():
    lw = _ConstantSource(200)
    png = render_gmgsi_tile(lw, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)
    assert rgba.shape == (32, 32, 4)
    assert rgba[0, 0, 0] == 200


def test_lw_only_renderer_below_threshold_is_transparent():
    """Below the current threshold (25, not the old 110): fully transparent.
    Just above it (30): nonzero, hand-computed truncated uint8 alpha.

    ramp = (30-25)/(140-25) = 5/115 ≈ 0.043478; alpha = ramp**0.7 ≈ 0.11138.
    ``_pack_rgba`` truncates alpha*255 to uint8 (not rounds):
    int(0.11138 * 255) == 28. Verified against the real render pipeline
    (this tile z/x/y sits well inside the disk, so the edge-feather
    multiplier is 1.0 and doesn't perturb the value — see module docstring).
    """
    lw = _ConstantSource(20)  # below current threshold (25)
    png = render_gmgsi_tile(lw, z=2, x=2, y=1, tile_size=32)
    rgba = _decode_png(png)
    assert np.all(rgba[..., 3] == 0)

    lw_above = _ConstantSource(30)  # just above current threshold
    png_above = render_gmgsi_tile(lw_above, z=2, x=2, y=1, tile_size=32)
    rgba_above = _decode_png(png_above)
    assert np.all(rgba_above[..., 3] == 28)
