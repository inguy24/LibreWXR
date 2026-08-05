# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Guard for D2: single-flight satellite warm with trailing rerun.

Concurrent ``warm_satellite()`` calls must coalesce into one live pass
plus at most one trailing pass — not N overlapping full multi-family
warms (the CPU-burn root cause: two overlapping ~60s warms per cycle).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from librewxr.tiles.warmer import TileWarmer

pytestmark = pytest.mark.tiles


async def test_warm_satellite_single_flight_with_trailing_rerun(monkeypatch):
    """Concurrent warm_satellite() calls coalesce; a call arriving mid-warm
    schedules exactly one trailing pass, not one pass per caller.

    Pre-change baseline has no ``_warm_satellite_once`` method — this
    guard fails with ``AttributeError`` there, a mechanics pin (not an
    assertion failure) confirming the guard exercises the D2 rename.
    """
    warmer = TileWarmer(
        store=MagicMock(),
        cache=MagicMock(),
        executor=MagicMock(),
    )

    running = 0
    max_concurrency = 0
    invocations = 0

    async def fake_warm_once():
        nonlocal running, max_concurrency, invocations
        running += 1
        max_concurrency = max(max_concurrency, running)
        invocations += 1
        await asyncio.sleep(0.02)
        running -= 1

    monkeypatch.setattr(warmer, "_warm_satellite_once", fake_warm_once)

    await asyncio.gather(
        warmer.warm_satellite(),
        warmer.warm_satellite(),
        warmer.warm_satellite(),
    )

    assert invocations == 2, (
        f"expected 1 live + 1 trailing pass, got {invocations} invocations"
    )
    assert max_concurrency == 1, (
        f"expected no overlapping passes, got max concurrency {max_concurrency}"
    )
