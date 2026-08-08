# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import overlapping_regions
from librewxr.tiles.renderer import compute_tile_geometry

logger = logging.getLogger(__name__)


class TileWarmer:
    """Pre-computes tile geometry for other timestamps when a cache miss occurs.

    Overview warming is demand-driven: past and nowcast tiles are only
    pre-computed after a user requests a corresponding frame type. The
    latest radar frame is pre-warmed at startup for fast initial load.

    What's cached is the pre-presentation ``TileGeometry`` — color
    scheme, output format, and arrow style are applied per-request in
    ``present_tile``.  That means the warmer's job is simpler than it
    used to be: warming one geometry per ``(ts, z, x, y, tile_size,
    smooth, snow)`` covers every color / format / arrow combination,
    which collapses what used to be ~18 warm jobs per tile into one.
    """

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        executor: ThreadPoolExecutor,
        enabled_regions: list[str] | None = None,
        nowcast_store=None,
        ecmwf_grid=None,
        nwp_chain=None,
        satellite_grids: dict[str, object] | None = None,
    ):
        self._store = store
        self._cache = cache
        self._executor = executor
        self._pending: dict[tuple, float] = {}
        self._lock = asyncio.Lock()
        self._pending_ttl = 300.0
        self._enabled_regions = enabled_regions
        self._nowcast_store = nowcast_store
        self._ecmwf_grid = ecmwf_grid
        self._nwp_chain = nwp_chain
        self._satellite_grids = satellite_grids
        self._past_warm_triggered = False
        self._nowcast_warm_triggered = False
        self._past_warm_complete = False
        self._nowcast_warm_complete = False
        self._warm_task: asyncio.Task | None = None
        self._satellite_warm_running: bool = False
        self._satellite_warm_rerun: bool = False

    def trigger_warm(self, frame_type: str = "both") -> None:
        """Trigger lazy overview warming from a user tile request.

        Sets the trigger flag so subsequent fetch cycles also warm this
        frame type.  Skips if a full pass for this type has already
        completed (new data from a fetch cycle resets the complete flag).
        """
        if frame_type in ("past", "both"):
            self._past_warm_triggered = True
        if frame_type in ("nowcast", "both"):
            self._nowcast_warm_triggered = True
        self._start_warm_if_needed()

    def schedule_warm(self) -> None:
        """Schedule overview warming for any triggered frame types.

        Called by the fetcher after each fetch cycle.  Resets the
        complete flags so new data gets warmed, then starts a warm
        pass if any frame type has been triggered.
        """
        self._past_warm_complete = False
        self._nowcast_warm_complete = False
        self._start_warm_if_needed()

    def _start_warm_if_needed(self) -> None:
        if self._warm_task is not None and not self._warm_task.done():
            return
        need_past = self._past_warm_triggered and not self._past_warm_complete
        need_nowcast = self._nowcast_warm_triggered and not self._nowcast_warm_complete
        if not need_past and not need_nowcast:
            return
        if need_past and need_nowcast:
            frame_type = "both"
        elif need_past:
            frame_type = "past"
        else:
            frame_type = "nowcast"
        self._warm_task = asyncio.create_task(
            self.warm_overview(frame_type=frame_type)
        )

    async def warm_latest(self) -> None:
        """Pre-compute overview geometry for the latest radar frame.

        Called at startup so the initial map load is fast.  Only warms
        past (radar) tiles — nowcast overview is triggered on demand.
        """
        max_zoom = settings.warm_overview_zoom
        max_zoom_regional = settings.warm_overview_zoom_regional
        max_zoom_total = max(max_zoom, max_zoom_regional)
        if max_zoom_total < 0:
            return

        timestamps = await self._store.get_timestamps()
        if not timestamps:
            return
        latest_ts = max(timestamps)
        frame = await self._store.get_frame(latest_ts)
        if frame is None:
            return

        tiles_by_zoom = self._build_tile_lists(max_zoom, max_zoom_regional, max_zoom_total, self._enabled_regions)
        total_tiles = sum(len(tiles) for tiles in tiles_by_zoom.values())
        logger.info("Warming latest frame ts=%d (%d tiles)", latest_ts, total_tiles)

        loop = asyncio.get_running_loop()
        frame_regions = frame.regions
        processed = 0
        submitted = 0
        next_pct = 10
        start = time.monotonic()
        for z in range(max_zoom_total + 1):
            for x, y in tiles_by_zoom[z]:
                processed += 1
                if processed % 500 == 0:
                    await asyncio.sleep(0)
                cache_key = (latest_ts, z, x, y, 256, False, False)
                if self._cache.get(cache_key) is not None:
                    continue
                async with self._lock:
                    if cache_key in self._pending:
                        if time.monotonic() - self._pending[cache_key] < self._pending_ttl:
                            continue
                    self._pending[cache_key] = time.monotonic()
                self._submit_compute(
                    loop, cache_key, frame_regions,
                    z, x, y, 256, False, False,
                    latest_ts, None,
                )
                submitted += 1
                pct = processed * 100 // total_tiles
                if pct >= next_pct or processed == total_tiles:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "Warm latest: %d/%d (%d%%), %d submitted, %.1fs",
                        processed, total_tiles, pct, submitted, elapsed,
                    )
                    next_pct = ((pct // 10) + 1) * 10

        elapsed = time.monotonic() - start
        logger.info(
            "Warm latest complete: %d submitted, %d skipped, %.1fs elapsed",
            submitted, processed - submitted, elapsed,
        )

    async def warm(
        self,
        triggered_timestamp: int,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        smooth: bool,
        snow: bool,
        frame_type: str = "past",
    ) -> None:
        """Schedule background geometry computes for all other timestamps.

        Also triggers overview warming for the given frame type so that
        demand-driven pre-warming kicks in after the first cache miss.
        """
        self.trigger_warm(frame_type)

        timestamps = await self._store.get_timestamps()
        nowcast_timestamps: set[int] = set()
        if self._nowcast_store is not None:
            nc_ts = await self._nowcast_store.get_timestamps()
            nowcast_timestamps = set(nc_ts)
            timestamps = list(set(timestamps) | nowcast_timestamps)
            timestamps.sort(reverse=True)

        loop = asyncio.get_running_loop()

        for ts in timestamps:
            if ts == triggered_timestamp:
                continue

            cache_key = (ts, z, x, y, tile_size, smooth, snow)

            if self._cache.get(cache_key) is not None:
                continue

            async with self._lock:
                if cache_key in self._pending:
                    if time.monotonic() - self._pending[cache_key] < self._pending_ttl:
                        continue
                self._pending[cache_key] = time.monotonic()

            nowcast_blend = None
            frame = await self._store.get_frame(ts)
            if frame is None and self._nowcast_store is not None:
                nc_frame, nowcast_blend = await self._nowcast_store.get_frame(ts)
                if nc_frame is not None:
                    frame = nc_frame
            if frame is None:
                async with self._lock:
                    self._pending.pop(cache_key, None)
                continue

            frame_regions = frame.regions
            self._submit_compute(
                loop, cache_key, frame_regions,
                z, x, y, tile_size, smooth, snow,
                ts, nowcast_blend,
            )

    @staticmethod
    def _build_tile_lists(
        max_zoom: int,
        max_zoom_regional: int,
        max_zoom_total: int,
        enabled_regions: list[str] | None,
    ) -> dict[int, list[tuple[int, int]]]:
        # Every zoom level warms only tiles overlapping an enabled region.
        # Tiles with no region overlap render as fully transparent — cheap
        # to produce on demand, so pre-building them (and permanently
        # caching their per-tile coordinate grids) is pure waste.  With a
        # BBOX configured the regions are already cropped to it, so this
        # bounds the warm set to the operator's coverage area at all zooms.
        tiles_by_zoom: dict[int, list[tuple[int, int]]] = {}
        for z in range(max_zoom_total + 1):
            n = 2**z
            tiles_by_zoom[z] = [
                (x, y) for y in range(n) for x in range(n)
                if overlapping_regions(z, x, y, enabled_regions)
            ]
        return tiles_by_zoom

    async def warm_overview(
        self,
        frame_type: str = "both",
        max_zoom: int | None = None,
        max_zoom_regional: int | None = None,
        tile_size: int = 256,
        smooth: bool = False,
        snow: bool = False,
    ) -> None:
        """Pre-compute overview geometry for every timestamp.

        ``frame_type`` controls which timestamps are warmed:
        - "past": only radar store timestamps
        - "nowcast": only nowcast store timestamps
        - "both": all timestamps (default)

        Two zoom passes, both restricted to tiles whose bbox overlaps an
        enabled region:
        - Zooms 0..max_zoom (overview levels).
        - Zooms max_zoom+1..max_zoom_regional (regional levels).
        """
        if max_zoom is None:
            max_zoom = settings.warm_overview_zoom
        if max_zoom_regional is None:
            max_zoom_regional = settings.warm_overview_zoom_regional

        past_ts: list[int] = []
        nowcast_ts: list[int] = []
        if frame_type in ("past", "both"):
            past_ts = await self._store.get_timestamps()
        if frame_type in ("nowcast", "both") and self._nowcast_store is not None:
            nowcast_ts = await self._nowcast_store.get_timestamps()

        past_ts_set = set(past_ts)
        nowcast_ts_set = set(nowcast_ts)
        timestamps = list(past_ts_set | nowcast_ts_set)
        timestamps.sort(reverse=True)

        max_zoom_total = max(max_zoom, max_zoom_regional)
        if max_zoom_total < 0 or not timestamps:
            if frame_type in ("past", "both"):
                self._past_warm_complete = True
            if frame_type in ("nowcast", "both"):
                self._nowcast_warm_complete = True
            return

        tiles_by_zoom = self._build_tile_lists(
            max_zoom, max_zoom_regional, max_zoom_total, self._enabled_regions,
        )

        loop = asyncio.get_running_loop()
        tiles_per_ts = sum(len(tiles) for tiles in tiles_by_zoom.values())
        total_tiles = tiles_per_ts * len(timestamps)
        logger.info(
            "Starting warm_overview type=%s (%d timestamps, %d tiles)",
            frame_type, len(timestamps), total_tiles,
        )
        processed = 0
        submitted = 0
        next_pct = 10
        start = time.monotonic()
        for ts in timestamps:
            nowcast_blend = None
            frame = None
            if ts in past_ts_set:
                frame = await self._store.get_frame(ts)
            if frame is None and ts in nowcast_ts_set and self._nowcast_store is not None:
                nc_frame, nowcast_blend = await self._nowcast_store.get_frame(ts)
                if nc_frame is not None:
                    frame = nc_frame
            if frame is None:
                continue
            frame_regions = frame.regions

            for z in range(max_zoom_total + 1):
                for x, y in tiles_by_zoom[z]:
                    processed += 1
                    if processed % 500 == 0:
                        await asyncio.sleep(0)
                    cache_key = (ts, z, x, y, tile_size, smooth, snow)
                    if self._cache.get(cache_key) is not None:
                        continue
                    async with self._lock:
                        if cache_key in self._pending:
                            if time.monotonic() - self._pending[cache_key] < self._pending_ttl:
                                continue
                        self._pending[cache_key] = time.monotonic()
                    self._submit_compute(
                        loop, cache_key, frame_regions,
                        z, x, y, tile_size, smooth, snow,
                        ts, nowcast_blend,
                    )
                    submitted += 1
                    pct = processed * 100 // total_tiles
                    if pct >= next_pct or processed == total_tiles:
                        elapsed = time.monotonic() - start
                        logger.info(
                            "Warm overview: %d/%d (%d%%), %d submitted, %.1fs",
                            processed, total_tiles, pct, submitted, elapsed,
                        )
                        next_pct = ((pct // 10) + 1) * 10

        elapsed = time.monotonic() - start
        if submitted > 0:
            logger.info(
                "Warm overview complete type=%s: %d submitted, %d skipped, %.1fs elapsed",
                frame_type, submitted, processed - submitted, elapsed,
            )
        else:
            logger.debug(
                "Warm overview complete type=%s: all %d tiles already cached, %.1fs elapsed",
                frame_type, processed, elapsed,
            )
        if frame_type in ("past", "both"):
            self._past_warm_complete = True
        if frame_type in ("nowcast", "both"):
            self._nowcast_warm_complete = True

    async def warm_satellite_demand(
        self,
        triggered_timestamp: int,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        fmt: str,
    ) -> None:
        """Pre-render satellite tiles at (z,x,y) for all other timestamps.

        Triggered on satellite tile cache miss — same demand-driven pattern
        as warm() for radar tiles.
        """
        if not self._satellite_grids:
            return

        from librewxr.sources.satellite._geo_base import GeoSatSource
        from librewxr.tiles.satellite_renderer import (
            render_geo_satellite_tile,
            render_gmgsi_composite_tile,
            render_gmgsi_tile,
            render_multi_satellite_tile,
        )

        # Build source index — same logic as warm_satellite()
        by_family: dict[str, dict[str, object]] = {}
        for slug, grid in self._satellite_grids.items():
            if grid is None or not bool(grid.timestamps):
                continue
            if slug.endswith("_ir_grid") or slug.endswith("_lw_grid"):
                if slug.endswith("_ir_grid"):
                    family = slug[: -len("_ir_grid")]
                else:
                    family = slug[: -len("_lw_grid")]
                by_family.setdefault(family, {})["ir"] = grid
            elif slug.endswith("_vis_grid"):
                family = slug[: -len("_vis_grid")]
                by_family.setdefault(family, {})["vis"] = grid

        multi_geo_sources: list[tuple[object, object | None, float]] = []
        all_family_names: list[str] = []
        for family, channels in sorted(by_family.items()):
            if "ir" in channels and isinstance(channels["ir"], GeoSatSource):
                multi_geo_sources.append(
                    (channels["ir"], channels.get("vis"), channels["ir"].sat_lon)
                )
                all_family_names.append(family)

        use_multi = len(multi_geo_sources) > 1

        ir_source = None
        vis_source = None
        is_geo = False
        for family, channels in by_family.items():
            if "ir" in channels:
                ir_source = channels["ir"]
                vis_source = channels.get("vis")
                is_geo = isinstance(ir_source, GeoSatSource)
                break

        if ir_source is None and not use_multi:
            return

        if use_multi:
            families_tag = "+".join(sorted(all_family_names))
            all_ts: set[int] = set()
            for ir_src, _, _ in multi_geo_sources:
                all_ts.update(ir_src.timestamps)
            timestamps = sorted(all_ts)
        else:
            timestamps = list(ir_source.timestamps)

        if not timestamps:
            return

        loop = asyncio.get_running_loop()
        submitted = 0

        for ts in timestamps:
            if ts == triggered_timestamp:
                continue

            if use_multi:
                cache_key = ("sat", "multi", families_tag, ts, z, x, y, tile_size, fmt)
            else:
                backing = "composite" if vis_source is not None else "ir_only"
                cache_key = ("sat", backing, ts, z, x, y, tile_size, fmt)

            if self._cache.get(cache_key) is not None:
                continue

            if use_multi:
                render_fn = render_multi_satellite_tile
                kwargs = dict(
                    sources=multi_geo_sources,
                    z=z, x=x, y=y,
                    timestamp=ts,
                    tile_size=tile_size,
                    fmt=fmt,
                )
            elif is_geo:
                render_fn = render_geo_satellite_tile
                kwargs = dict(
                    ir_source=ir_source,
                    vis_source=vis_source,
                    z=z, x=x, y=y,
                    tile_size=tile_size,
                    timestamp=ts,
                    fmt=fmt,
                )
            elif vis_source is not None:
                render_fn = render_gmgsi_composite_tile
                kwargs = dict(
                    lw_source=ir_source,
                    vis_source=vis_source,
                    z=z, x=x, y=y,
                    tile_size=tile_size,
                    timestamp=ts,
                    fmt=fmt,
                )
            else:
                render_fn = render_gmgsi_tile
                kwargs = dict(
                    source=ir_source,
                    z=z, x=x, y=y,
                    tile_size=tile_size,
                    timestamp=ts,
                    fmt=fmt,
                )

            try:
                tile_bytes = await loop.run_in_executor(
                    self._executor,
                    lambda fn=render_fn, kw=kwargs: fn(**kw),
                )
                self._cache.put(cache_key, tile_bytes)
                submitted += 1
            except Exception:
                logger.debug(
                    "Satellite demand warm failed for z=%d x=%d y=%d ts=%d", z, x, y, ts
                )

            await asyncio.sleep(0)

        logger.debug(
            "Satellite demand warm complete: z=%d x=%d y=%d triggered_ts=%d, %d submitted",
            z, x, y, triggered_timestamp, submitted,
        )

    async def warm_satellite(self) -> None:
        """Coalesce concurrent satellite warm requests into a single pass.

        Concurrent calls coalesce; a call arriving mid-warm schedules
        exactly one trailing pass so frames ingested by the second family
        after the first pass indexed its grids are still warmed (the
        trailing pass is mostly cache-skips). Single-threaded asyncio:
        plain bools are safe, no lock needed. If the in-flight pass
        raises, a queued trailing rerun is deferred to the next ingest
        trigger (every fetch cycle) rather than retried immediately.
        """
        if self._satellite_warm_running:
            self._satellite_warm_rerun = True
            return
        self._satellite_warm_running = True
        try:
            while True:
                self._satellite_warm_rerun = False
                await self._warm_satellite_once()
                if not self._satellite_warm_rerun:
                    break
        finally:
            self._satellite_warm_running = False

    async def _warm_satellite_once(self) -> None:
        """Pre-render satellite tiles at overview zoom levels.

        Iterates satellite timestamps x overview tile coordinates, renders
        each tile using the appropriate renderer (opaque for GeoSat, semi-
        transparent for GMGSI), and stores the result in the tile cache.

        When multiple geostationary families are loaded, uses the multi-
        satellite renderer for seamless compositing across satellite disks.
        """
        if not self._satellite_grids:
            return

        from librewxr.sources.satellite._geo_base import GeoSatSource
        from librewxr.tiles.satellite_renderer import (
            render_geo_satellite_tile,
            render_gmgsi_composite_tile,
            render_gmgsi_tile,
            render_multi_satellite_tile,
        )

        # Index all families (same logic as routes._find_all_satellite_families)
        ir_source = None
        vis_source = None
        by_family: dict[str, dict[str, object]] = {}
        for slug, grid in self._satellite_grids.items():
            if grid is None or not bool(grid.timestamps):
                continue
            if slug.endswith("_ir_grid") or slug.endswith("_lw_grid"):
                if slug.endswith("_ir_grid"):
                    family = slug[: -len("_ir_grid")]
                else:
                    family = slug[: -len("_lw_grid")]
                by_family.setdefault(family, {})["ir"] = grid
            elif slug.endswith("_vis_grid"):
                family = slug[: -len("_vis_grid")]
                by_family.setdefault(family, {})["vis"] = grid

        # Build multi-satellite source list for geostationary families
        multi_geo_sources: list[tuple[object, object | None, float]] = []
        all_family_names: list[str] = []
        for family, channels in sorted(by_family.items()):
            if "ir" in channels and isinstance(channels["ir"], GeoSatSource):
                multi_geo_sources.append(
                    (channels["ir"], channels.get("vis"), channels["ir"].sat_lon)
                )
                all_family_names.append(family)

        use_multi = len(multi_geo_sources) > 1

        # Also find single best source for fallback / single-family path
        for family, channels in by_family.items():
            if "ir" in channels:
                ir_source = channels["ir"]
                vis_source = channels.get("vis")
                break

        if ir_source is None and not use_multi:
            return

        if use_multi:
            families_tag = "+".join(sorted(all_family_names))
            logger.info("Satellite warm: multi-family mode (%s)", families_tag)
            # Use union of all source timestamps
            all_ts: set[int] = set()
            for ir_src, _, _ in multi_geo_sources:
                all_ts.update(ir_src.timestamps)
            timestamps = sorted(all_ts)
        else:
            is_geo = isinstance(ir_source, GeoSatSource)
            timestamps = list(ir_source.timestamps)

        if not timestamps:
            return

        max_zoom = settings.warm_overview_zoom
        max_zoom_regional = settings.warm_overview_zoom_regional
        max_zoom_total = max(max_zoom, max_zoom_regional)
        if max_zoom_total < 0:
            return

        tiles_by_zoom = self._build_tile_lists(
            max_zoom, max_zoom_regional, max_zoom_total, self._enabled_regions,
        )

        tile_size = 512
        fmt = "webp"
        submitted = 0
        skipped = 0
        start = time.monotonic()

        loop = asyncio.get_running_loop()

        for ts in timestamps:
            for z in range(max_zoom_total + 1):
                for x, y in tiles_by_zoom[z]:
                    if use_multi:
                        cache_key = ("sat", "multi", families_tag, ts, z, x, y, tile_size, fmt)
                    else:
                        backing = "composite" if vis_source is not None else "ir_only"
                        cache_key = ("sat", backing, ts, z, x, y, tile_size, fmt)
                    if self._cache.get(cache_key) is not None:
                        skipped += 1
                        continue

                    if use_multi:
                        render_fn = render_multi_satellite_tile
                        kwargs = dict(
                            sources=multi_geo_sources,
                            z=z, x=x, y=y,
                            timestamp=ts,
                            tile_size=tile_size,
                            fmt=fmt,
                        )
                    elif is_geo:
                        render_fn = render_geo_satellite_tile
                        kwargs = dict(
                            ir_source=ir_source,
                            vis_source=vis_source,
                            z=z, x=x, y=y,
                            tile_size=tile_size,
                            timestamp=ts,
                            fmt=fmt,
                        )
                    elif vis_source is not None:
                        render_fn = render_gmgsi_composite_tile
                        kwargs = dict(
                            lw_source=ir_source,
                            vis_source=vis_source,
                            z=z, x=x, y=y,
                            tile_size=tile_size,
                            timestamp=ts,
                            fmt=fmt,
                        )
                    else:
                        render_fn = render_gmgsi_tile
                        kwargs = dict(
                            source=ir_source,
                            z=z, x=x, y=y,
                            tile_size=tile_size,
                            timestamp=ts,
                            fmt=fmt,
                        )

                    try:
                        tile_bytes = await loop.run_in_executor(
                            self._executor,
                            lambda fn=render_fn, kw=kwargs: fn(**kw),
                        )
                        self._cache.put(cache_key, tile_bytes)
                        submitted += 1
                    except Exception:
                        logger.debug("Satellite warm failed for z=%d x=%d y=%d ts=%d", z, x, y, ts)

                    if (submitted + skipped) % 200 == 0:
                        await asyncio.sleep(0)

        elapsed = time.monotonic() - start
        logger.info(
            "Satellite warm: %d submitted, %d skipped (cached), %.1fs elapsed",
            submitted, skipped, elapsed,
        )

    def _compute_and_cache(
        self,
        cache_key: tuple,
        frame_regions: dict,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        smooth: bool,
        snow: bool,
        frame_timestamp: int | None = None,
        nowcast_blend: float | None = None,
    ) -> None:
        """Compute tile geometry and store it in the cache (runs in thread pool)."""
        try:
            if self._cache.get(cache_key) is not None:
                return
            geom = compute_tile_geometry(
                frame_regions=frame_regions,
                z=z, x=x, y=y,
                tile_size=tile_size,
                smooth=smooth,
                snow=snow,
                nwp_chain=self._nwp_chain,
                enabled_regions=self._enabled_regions,
                frame_timestamp=frame_timestamp,
                nowcast_blend=nowcast_blend,
            )
            self._cache.put(cache_key, geom)
        except Exception:
            logger.debug("Warm compute failed for key %s", cache_key[:5])
        finally:
            self._pending.pop(cache_key, None)

    def _submit_compute(
        self,
        loop: asyncio.AbstractEventLoop,
        cache_key: tuple,
        frame_regions: dict,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        smooth: bool,
        snow: bool,
        ts: int,
        nowcast_blend: float | None,
    ) -> None:
        """Schedule a compute on the executor with exception logging."""
        future = loop.run_in_executor(
            self._executor,
            self._compute_and_cache,
            cache_key,
            frame_regions,
            z, x, y,
            tile_size,
            smooth,
            snow,
            ts,
            nowcast_blend,
        )
        future.add_done_callback(self._log_compute_exception)

    @staticmethod
    def _log_compute_exception(future) -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.warning("Warm compute task raised: %r", exc)

    def shutdown(self) -> None:
        pass  # Executor is shared; lifecycle managed by main.py
