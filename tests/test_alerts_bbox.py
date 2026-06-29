# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for alert BBOX filtering.

The WMOAlertsFetcher._fetch_once method filters alerts by BBOX after
ingest: alerts whose polygon intersects the BBOX pass, alerts fully
outside the BBOX are dropped, and alerts without polygon geometry are
also dropped (since their relevance cannot be determined).

These tests exercise the filter logic directly via Shapely, matching
the exact predicate used in _fetch_once (polygon.intersects(bbox_poly)).
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon, box

from librewxr.data.alerts_store import AlertEntry

pytestmark = pytest.mark.alerts


def _make_alert(polygon: Polygon | None, event: str = "Test") -> AlertEntry:
    return AlertEntry(
        source_id="test",
        event=event,
        description="test alert",
        severity="Moderate",
        effective="2026-06-28T00:00:00Z",
        expires="2026-06-29T00:00:00Z",
        area_desc="Test Area",
        url="https://example.com/alert",
        polygon=polygon,
    )


def _apply_bbox_filter(
    alerts: list[AlertEntry],
    bbox: tuple[float, float, float, float] | None,
) -> list[AlertEntry]:
    """Replicate the BBOX filter from WMOAlertsFetcher._fetch_once."""
    if bbox is None:
        return alerts
    south, west, north, east = bbox
    bbox_poly = box(west, south, east, north)
    return [
        a for a in alerts
        if a.polygon is not None and a.polygon.intersects(bbox_poly)
    ]


class TestAlertBBOXFilter:
    """Alert BBOX geographic filtering."""

    def test_inside_bbox_passes(self):
        """Alert polygon fully inside BBOX passes the filter."""
        bbox = (30.0, -125.0, 50.0, -70.0)  # broad US
        poly = Polygon([(-100, 35), (-99, 35), (-99, 36), (-100, 36), (-100, 35)])
        alerts = [_make_alert(poly, "Inside")]
        result = _apply_bbox_filter(alerts, bbox)
        assert len(result) == 1
        assert result[0].event == "Inside"

    def test_outside_bbox_rejected(self):
        """Alert polygon fully outside BBOX is rejected."""
        bbox = (32.0, -120.5, 35.5, -114.5)  # SoCal
        # Polygon in Europe
        poly = Polygon([(10, 48), (11, 48), (11, 49), (10, 49), (10, 48)])
        alerts = [_make_alert(poly, "Outside")]
        result = _apply_bbox_filter(alerts, bbox)
        assert len(result) == 0

    def test_bisecting_bbox_passes(self):
        """Alert polygon crossing the BBOX boundary passes (intersects)."""
        bbox = (32.0, -120.5, 35.5, -114.5)  # SoCal
        # Polygon straddles the western edge
        poly = Polygon([
            (-121.0, 33.0), (-119.0, 33.0),
            (-119.0, 34.0), (-121.0, 34.0), (-121.0, 33.0),
        ])
        alerts = [_make_alert(poly, "Bisecting")]
        result = _apply_bbox_filter(alerts, bbox)
        assert len(result) == 1
        assert result[0].event == "Bisecting"

    def test_no_bbox_passes_all(self):
        """With no BBOX configured, all alerts pass unfiltered."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        alerts = [
            _make_alert(poly, "With poly"),
            _make_alert(None, "No poly"),
        ]
        result = _apply_bbox_filter(alerts, None)
        assert len(result) == 2

    def test_null_polygon_dropped_with_bbox(self):
        """Alert without polygon geometry is dropped when BBOX is set."""
        bbox = (32.0, -120.5, 35.5, -114.5)
        alerts = [_make_alert(None, "No geometry")]
        result = _apply_bbox_filter(alerts, bbox)
        assert len(result) == 0

    def test_mixed_alerts_filtered_correctly(self):
        """Mix of inside, outside, bisecting, and null-polygon alerts."""
        bbox = (32.0, -120.5, 35.5, -114.5)

        inside = Polygon([
            (-118, 33), (-117, 33), (-117, 34), (-118, 34), (-118, 33),
        ])
        outside = Polygon([
            (-80, 25), (-79, 25), (-79, 26), (-80, 26), (-80, 25),
        ])
        bisecting = Polygon([
            (-121, 34), (-119, 34), (-119, 36), (-121, 36), (-121, 34),
        ])

        alerts = [
            _make_alert(inside, "Inside"),
            _make_alert(outside, "Outside"),
            _make_alert(bisecting, "Bisecting"),
            _make_alert(None, "No polygon"),
        ]
        result = _apply_bbox_filter(alerts, bbox)
        events = {a.event for a in result}
        assert events == {"Inside", "Bisecting"}
