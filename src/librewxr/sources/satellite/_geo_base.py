# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared base for geostationary satellite sources (GOES, Himawari).

``GeoSatSource`` handles everything common to any geostationary imager
that publishes fixed-grid NetCDF files on anonymous S3:

- S3 listing + download via fsspec
- Frame retention ring buffer
- Geostationary projection for ``sample()``
- Disk cache / memmap / cross-worker pickle

Concrete subclasses (``GOESIRSource``, ``HimawariIRSource``, …) pin the
satellite-specific parameters (bucket, product path, band, sat_lon,
sat_height) and override ``_decode_netcdf`` for sensor-specific value
mapping.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import fsspec
import numpy as np
import xarray as xr

from librewxr.tiles.geostationary import forward as geo_forward

logger = logging.getLogger(__name__)

# Standard GEO orbital height (metres above ellipsoid) — shared by GOES
# and Himawari.  Individual sources override via their class var, but this
# constant is good enough for the coarse "is the BBOX visible?" check below.
_GEO_HEIGHT = 35786023.0


def bbox_overlaps_disk(
    bbox: tuple[float, float, float, float],
    sat_lon: float,
    sat_height: float = _GEO_HEIGHT,
) -> bool:
    """Check whether any part of a BBOX is visible to a geostationary satellite.

    Tests the four corners plus edge midpoints against the satellite's
    forward projection.  Returns True if at least one test point is on the
    visible disk (non-NaN scan angles).  Used by satellite providers to
    decide whether to enable a family for the operator's BBOX.
    """
    south, west, north, east = bbox
    mid_lat = (south + north) / 2.0
    mid_lon = (west + east) / 2.0
    test_lats = np.array([
        south, south, north, north, mid_lat,
        south, north, mid_lat, mid_lat,
    ])
    test_lons = np.array([
        west, east, west, east, mid_lon,
        mid_lon, mid_lon, west, east,
    ])
    x_ang, y_ang = geo_forward(
        test_lats, test_lons, sat_lon, sat_height,
    )
    return bool(np.any(~(np.isnan(x_ang) | np.isnan(y_ang))))


class GeoSatSource:
    """Abstract base for one channel of a geostationary satellite."""

    # ── Subclass-defined class variables ──

    # Satellite orbital parameters
    sat_lon: ClassVar[float]  # sub-satellite longitude (degrees)
    sat_height: ClassVar[float]  # orbital height above ellipsoid (metres)

    # S3 addressing
    s3_bucket: ClassVar[str]
    s3_product_path: ClassVar[str]  # e.g. "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str]  # e.g. "CMIPC-M6C13" for filtering listings

    # Display
    friendly_name: ClassVar[str]
    channel: ClassVar[str]  # "IR" or "VIS"

    # Cadence in minutes (5 for GOES CONUS, 10 for Himawari full-disk)
    cadence_minutes: ClassVar[int] = 5

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_frames: int = 36,
        bbox: tuple[float, float, float, float] | None = None,
        downsample_factor: int = 1,
        cadence_override: int = 0,
    ) -> None:
        self.name = self.friendly_name
        self._frames: dict[int, np.ndarray] = {}
        self._sorted_timestamps: list[int] = []
        # ts -> the S3 key that produced the stored frame.  Not persisted
        # to the snapshot/disk cache (see __setstate__ / _load_cached_frames);
        # a frame resident from a restore has no entry here until adopted.
        self._frame_keys: dict[int, str] = {}
        # Timestamps whose frame was replaced by a newer reprocessed scan
        # since the last consume_replaced_timestamps() call.
        self._replaced_timestamps: list[int] = []
        self._fs: fsspec.AbstractFileSystem | None = None
        self._max_frames = max_frames
        self._bbox = bbox
        self._downsample_factor = downsample_factor
        self._effective_cadence_minutes = (
            cadence_override // 60 if cadence_override > 0
            else self.cadence_minutes
        )

        # Per-frame grid metadata (set on first decode)
        self._x_vec: np.ndarray | None = None  # 1-D scan-angle x coords
        self._y_vec: np.ndarray | None = None  # 1-D scan-angle y coords
        self._grid_height: int = 0  # post-crop/downsample (used by sample())
        self._grid_width: int = 0
        self._full_grid_height: int = 0  # raw NetCDF dims (used by shape check)
        self._full_grid_width: int = 0

        # BBOX crop indices (computed once after first grid init)
        self._crop_row_start: int = 0
        self._crop_row_end: int = 0
        self._crop_col_start: int = 0
        self._crop_col_end: int = 0
        self._crop_computed: bool = False

        self._cache_root: Path | None = (
            Path(cache_dir) if cache_dir else None
        )
        self._channel_cache_dir: Path | None = None
        if self._cache_root is not None:
            self._channel_cache_dir = (
                self._cache_root / self._cache_subdir() / self.channel
            )
            self._channel_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cached_frames()

    def _cache_subdir(self) -> str:
        """Subdirectory name under the cache root (e.g. ``goes18``)."""
        return self.s3_bucket.replace("-", "_").replace("noaa_", "")

    # ── Public state ──

    @property
    def timestamps(self) -> list[int]:
        return list(self._sorted_timestamps)

    @property
    def loaded(self) -> bool:
        return bool(self._sorted_timestamps)

    @property
    def data_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._frames.values())

    # ── Fetch / decode ──

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3", anon=True, listings_expiry_time=120,
            )
        return self._fs

    async def fetch(self) -> bool:
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception:
            logger.exception("%s: fetch failed", self.friendly_name)
            return False

    def _fetch_sync(self) -> bool:
        """List, dedupe/trim, and ingest new or reprocessed frames.

        Per-timestamp key choice is latest-wins: NOAA zero-pads the ``_c``
        creation token, so the lexicographically largest key for a given
        timestamp is the most recently created (and, for a republish, the
        reprocessed/corrected product). A resident timestamp whose newly
        listed key is newer than the one on record is re-downloaded and
        replaces the stored frame (see ``consume_replaced_timestamps``).
        A resident timestamp with no recorded key (restored from the disk
        cache / cross-worker snapshot, where ``_frame_keys`` isn't
        persisted) silently adopts the listed key instead of
        re-downloading — this also means a republish that happened
        entirely while this process was down is accepted as missed rather
        than retroactively detected.
        """
        fs = self._get_fs()
        now = datetime.now(timezone.utc)
        window_hours = max(1, (self._max_frames * self._effective_cadence_minutes) // 60 + 1)
        window_start = now - timedelta(hours=window_hours)
        keys = self._list_recent_keys(fs, window_start, now)
        if not keys:
            logger.warning("%s: no S3 keys in retention window", self.friendly_name)
            return False

        # Any key beyond the newest max_frames would be evicted by the trim
        # loop immediately after ingest, so downloading it is pure waste.
        # The generous listing window above is still needed to refill the
        # store after restarts/gaps — window_hours is unchanged.
        # Dedupe by timestamp BEFORE trimming, latest key per timestamp
        # winning (see method docstring), so a republish doesn't consume
        # an extra retention slot and push out a genuinely newer distinct
        # timestamp.
        best_key_by_ts: dict[int, str] = {}
        for unix_ts, s3_key in sorted(keys):
            best_key_by_ts[unix_ts] = s3_key
        keys = sorted(best_key_by_ts.items())[-self._max_frames :]

        new_count = 0
        for unix_ts, s3_key in keys:
            if unix_ts not in self._frames:
                arr = self._download_and_decode(fs, s3_key)
                if arr is None:
                    continue
                self._frames[unix_ts] = arr
                self._frame_keys[unix_ts] = s3_key
                new_count += 1
                if self._channel_cache_dir is not None:
                    self._write_cache(unix_ts, arr)
                continue

            recorded_key = self._frame_keys.get(unix_ts)
            if recorded_key is None:
                # Resident from a disk-cache/snapshot restore — key
                # unknown. Adopt the listed key without re-downloading.
                self._frame_keys[unix_ts] = s3_key
                continue
            if s3_key <= recorded_key:
                continue  # already have this (or an older) key

            arr = self._download_and_decode(fs, s3_key)
            if arr is None:
                continue
            self._frames[unix_ts] = arr
            self._frame_keys[unix_ts] = s3_key
            # Record the replacement the moment the in-memory frame changes,
            # BEFORE the disk-cache write: once _frame_keys has advanced, no
            # future poll can re-detect this replacement, so an exception
            # later in this call (e.g. a failing _write_cache) must not be
            # able to lose the pending tile-cache invalidation.
            self._replaced_timestamps.append(unix_ts)
            if self._channel_cache_dir is not None:
                self._write_cache(unix_ts, arr)
            logger.info(
                "%s: replaced reprocessed frame ts=%d (%s)",
                self.friendly_name, unix_ts, s3_key,
            )

        self._sorted_timestamps = sorted(self._frames)
        while len(self._sorted_timestamps) > self._max_frames:
            oldest = self._sorted_timestamps.pop(0)
            self._frames.pop(oldest, None)
            self._frame_keys.pop(oldest, None)
            if self._channel_cache_dir is not None:
                self._cache_path_for(oldest).unlink(missing_ok=True)

        if new_count:
            logger.info(
                "%s: ingested %d new frame(s); store holds %d",
                self.friendly_name, new_count, len(self._sorted_timestamps),
            )
        # Pending (unconsumed) replacements force the True path even when
        # nothing new was ingested this poll: if a prior poll recorded a
        # replacement and then raised before the fetcher could consume it,
        # the next clean poll must still trigger the fetcher's
        # consume-and-invalidate step.
        return new_count > 0 or bool(self._replaced_timestamps)

    def consume_replaced_timestamps(self) -> list[int]:
        """Return and clear timestamps replaced by a newer reprocessed scan.

        Consumed by the fetcher to invalidate stale tile-cache entries for
        those timestamps after a successful fetch.
        """
        replaced = self._replaced_timestamps
        self._replaced_timestamps = []
        return replaced

    def _list_recent_keys(
        self,
        fs: fsspec.AbstractFileSystem,
        window_start: datetime,
        window_end: datetime,
    ) -> list[tuple[int, str]]:
        """List S3 keys for the retention window.

        Subclasses may override to handle different path layouts.
        Default walks GOES-style ``{product}/{year}/{doy}/{hour}/``.
        """
        results: list[tuple[int, str]] = []
        cursor = window_start.replace(minute=0, second=0, microsecond=0)
        while cursor <= window_end:
            doy = cursor.timetuple().tm_yday
            prefix = (
                f"{self.s3_bucket}/{self.s3_product_path}/"
                f"{cursor.year:04d}/{doy:03d}/{cursor.hour:02d}/"
            )
            try:
                entries = fs.ls(prefix, detail=False)
            except FileNotFoundError:
                entries = []
            except Exception:
                logger.exception("%s: failed to list %s", self.friendly_name, prefix)
                entries = []
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                if self.s3_filename_token not in name:
                    continue
                unix_ts = self._parse_start_timestamp(name)
                if unix_ts is None:
                    continue
                results.append((unix_ts, entry))
            cursor += timedelta(hours=1)
        results = sorted(set(results))
        return results

    @staticmethod
    def _parse_start_timestamp(filename: str) -> int | None:
        """Parse ``_s{YYYYDDDHHMMSSt}`` token to Unix timestamp.

        GOES/Himawari filenames use day-of-year format, unlike GMGSI's
        ``YYYYMMDDHHMMSS``.  We keep minute-level precision (not floored
        to the hour) since the cadence is 5–10 minutes.
        """
        try:
            tok = filename.split("_s", 1)[1].split("_", 1)[0]
        except IndexError:
            return None
        if len(tok) < 13:
            return None
        try:
            yr = int(tok[0:4])
            doy = int(tok[4:7])
            hh = int(tok[7:9])
            mm = int(tok[9:11])
            ss = int(tok[11:13])
        except ValueError:
            return None
        try:
            dt = datetime(yr, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            return None
        return int(dt.timestamp())

    def _apply_post_decode(self, arr: np.ndarray) -> np.ndarray:
        """Apply BBOX crop and downsampling to a full-size decoded grid."""
        if self._crop_computed:
            arr = arr[
                self._crop_row_start:self._crop_row_end,
                self._crop_col_start:self._crop_col_end,
            ].copy()
        if self._downsample_factor > 1:
            f = self._downsample_factor
            h, w = arr.shape
            arr = (
                arr[: h - h % f, : w - w % f]
                .reshape(h // f, f, w // f, f)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )
        return arr

    def _download_and_decode(
        self, fs: fsspec.AbstractFileSystem, s3_key: str,
    ) -> np.ndarray | None:
        """Download and decode a satellite frame.

        When BBOX crop indices are established (after the first frame),
        streams only the BBOX region via HDF5 chunked reads.  The
        returned array is ready to store — crop and downsample are
        already applied for the streaming path.
        """
        if self._crop_computed and self._x_vec is not None:
            return self._stream_cropped(fs, s3_key)
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                fs.get(s3_key, tmp.name)
                arr = self._decode_netcdf(tmp.name)
                return self._apply_post_decode(arr) if arr is not None else None
        except Exception:
            logger.exception(
                "%s: download/decode failed for %s", self.friendly_name, s3_key,
            )
            return None

    def _stream_cropped(
        self, fs: fsspec.AbstractFileSystem, s3_key: str,
    ) -> np.ndarray | None:
        """Stream only the BBOX region from S3 via HDF5 chunked reads.

        After the first frame establishes grid vectors and crop indices,
        subsequent frames use h5py to read only the cropped slice
        directly from S3.  Only the HDF5 chunks that intersect the
        BBOX are downloaded — typically 2-5% of the full file.
        """
        import h5py

        try:
            with fs.open(s3_key, "rb") as f:
                with h5py.File(f, "r") as hf:
                    rs = self._crop_row_start
                    re = self._crop_row_end
                    cs = self._crop_col_start
                    ce = self._crop_col_end

                    raw = hf["CMI"][rs:re, cs:ce]
                    scale = hf["CMI"].attrs.get("scale_factor", None)
                    offset = hf["CMI"].attrs.get("add_offset", None)
                    if scale is not None or offset is not None:
                        cmi = raw.astype(np.float32)
                        if scale is not None:
                            cmi *= np.float32(scale)
                        if offset is not None:
                            cmi += np.float32(offset)
                    else:
                        cmi = raw.astype(np.float32)

                    if "DQF" in hf:
                        dqf = hf["DQF"][rs:re, cs:ce]
                        cmi = np.where(dqf == 0, cmi, np.nan)

                    arr = self._map_to_uint8(cmi)
                    if self._downsample_factor > 1:
                        f = self._downsample_factor
                        h, w = arr.shape
                        arr = (
                            arr[: h - h % f, : w - w % f]
                            .reshape(h // f, f, w // f, f)
                            .mean(axis=(1, 3))
                            .astype(np.uint8)
                        )
                    return arr
        except Exception:
            logger.exception(
                "%s: stream/decode failed for %s", self.friendly_name, s3_key,
            )
            return None

    @abstractmethod
    def _decode_netcdf(self, path: str) -> np.ndarray | None:
        """Open NetCDF, decode sensor-specific values, return uint8 grid.

        Must also set ``self._x_vec``, ``self._y_vec``, ``self._grid_height``,
        ``self._grid_width`` on first successful decode.
        """
        ...

    def _init_grid_vectors(self, ds: xr.Dataset) -> None:
        """Extract 1-D scan-angle coordinate vectors from the dataset.

        Only runs once — subsequent frames reuse the stored vectors
        (the fixed-grid coordinates never change for a given product).
        If a BBOX is configured, computes the crop slice in scan-angle
        space and trims the vectors to only the BBOX region.
        """
        if self._x_vec is not None:
            return
        full_x = ds["x"].values.astype(np.float64)
        full_y = ds["y"].values.astype(np.float64)
        full_h = len(full_y)
        full_w = len(full_x)

        self._full_grid_width = full_w
        self._full_grid_height = full_h

        if self._bbox is not None and not self._crop_computed:
            self._compute_crop_indices(full_x, full_y, full_w, full_h)

        if self._crop_computed:
            self._x_vec = full_x[self._crop_col_start:self._crop_col_end]
            self._y_vec = full_y[self._crop_row_start:self._crop_row_end]
        else:
            self._x_vec = full_x
            self._y_vec = full_y
        if self._downsample_factor > 1:
            f = self._downsample_factor
            self._x_vec = self._x_vec[f // 2 :: f]
            self._y_vec = self._y_vec[f // 2 :: f]
        self._grid_width = len(self._x_vec)
        self._grid_height = len(self._y_vec)

        if self._crop_computed:
            uncropped_kb = full_h * full_w / 1024
            cropped_kb = self._grid_height * self._grid_width / 1024
            logger.info(
                "%s: BBOX crop [%d:%d, %d:%d] → %d×%d (%.1f KB, was %.0f KB)",
                self.friendly_name,
                self._crop_row_start, self._crop_row_end,
                self._crop_col_start, self._crop_col_end,
                self._grid_height, self._grid_width,
                cropped_kb, uncropped_kb,
            )

    def _compute_crop_indices(
        self,
        full_x: np.ndarray,
        full_y: np.ndarray,
        full_w: int,
        full_h: int,
    ) -> None:
        """Compute row/col crop slice for the configured BBOX.

        Projects the BBOX corners to scan-angle space, finds the
        bounding scan-angle range, and maps that to array indices
        with a small margin for safety.
        """
        south, west, north, east = self._bbox
        corners_lat = np.array([south, south, north, north])
        corners_lon = np.array([west, east, west, east])

        x_ang, y_ang = geo_forward(
            corners_lat, corners_lon, self.sat_lon, self.sat_height,
        )
        visible = ~(np.isnan(x_ang) | np.isnan(y_ang))
        if not visible.any():
            return

        x_min = float(np.nanmin(x_ang[visible]))
        x_max = float(np.nanmax(x_ang[visible]))
        y_min = float(np.nanmin(y_ang[visible]))
        y_max = float(np.nanmax(y_ang[visible]))

        margin = 0.005  # ~0.3° extra on each side

        col_start = int(np.searchsorted(full_x, x_min - margin))
        col_end = int(np.searchsorted(full_x, x_max + margin, side="right"))
        col_start = max(0, col_start)
        col_end = min(full_w, col_end)

        # y_vec is typically descending (north to south)
        if full_y[0] > full_y[-1]:
            row_start = int(np.searchsorted(-full_y, -(y_max + margin)))
            row_end = int(np.searchsorted(-full_y, -(y_min - margin), side="right"))
        else:
            row_start = int(np.searchsorted(full_y, y_min - margin))
            row_end = int(np.searchsorted(full_y, y_max + margin, side="right"))
        row_start = max(0, row_start)
        row_end = min(full_h, row_end)

        if row_end <= row_start or col_end <= col_start:
            return

        self._crop_row_start = row_start
        self._crop_row_end = row_end
        self._crop_col_start = col_start
        self._crop_col_end = col_end
        self._crop_computed = True

    # ── Sampling ──

    def _nearest_timestamp(self, timestamp: int | None) -> int | None:
        if not self._sorted_timestamps:
            return None
        if timestamp is None:
            return self._sorted_timestamps[-1]
        ts_list = self._sorted_timestamps
        idx = np.searchsorted(ts_list, timestamp)
        if idx == 0:
            return ts_list[0]
        if idx >= len(ts_list):
            return ts_list[-1]
        before = ts_list[idx - 1]
        after = ts_list[idx]
        return before if timestamp - before <= after - timestamp else after

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Sample encoded uint8 values at the given lat/lon points.

        Uses the geostationary forward projection to convert lat/lon to
        scan angles, then nearest-neighbour lookup into the stored grid.
        Returns 0 (no data) for points not visible to the satellite.
        """
        out = np.zeros(lat.shape, dtype=np.uint8)
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            return out
        if self._x_vec is None or self._y_vec is None:
            return out

        grid = self._frames[ts]

        # Forward-project lat/lon to scan angles
        x_ang, y_ang = geo_forward(
            lat.astype(np.float64),
            lon.astype(np.float64),
            self.sat_lon,
            self.sat_height,
        )

        # Map scan angles to pixel indices via the stored coordinate vectors
        x_step = (self._x_vec[-1] - self._x_vec[0]) / (self._grid_width - 1)
        y_step = (self._y_vec[0] - self._y_vec[-1]) / (self._grid_height - 1)

        visible = ~(np.isnan(x_ang) | np.isnan(y_ang))
        x_safe = np.where(visible, x_ang, 0.0)
        y_safe = np.where(visible, y_ang, 0.0)
        col = ((x_safe - self._x_vec[0]) / x_step).astype(np.int32)
        row = ((self._y_vec[0] - y_safe) / y_step).astype(np.int32)

        in_bounds = (
            visible
            & (row >= 0) & (row < self._grid_height)
            & (col >= 0) & (col < self._grid_width)
        )

        row_safe = np.clip(row, 0, self._grid_height - 1)
        col_safe = np.clip(col, 0, self._grid_width - 1)

        sampled = grid[row_safe, col_safe]
        out = np.where(in_bounds, sampled, 0).astype(np.uint8)
        return out

    # ── Cache (disk persistence + cross-worker snapshot) ──

    def _cache_path_for(self, unix_ts: int) -> Path:
        assert self._channel_cache_dir is not None
        return self._channel_cache_dir / f"frame_{unix_ts}.dat"

    def _meta_cache_path(self) -> Path:
        assert self._channel_cache_dir is not None
        return self._channel_cache_dir / "grid_meta.npz"

    def _write_cache(self, unix_ts: int, arr: np.ndarray) -> None:
        final = self._cache_path_for(unix_ts)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(
            tmp, dtype=np.uint8, mode="w+",
            shape=arr.shape,
        )
        mm[:] = arr
        mm.flush()
        del mm
        os.replace(tmp, final)
        # Persist grid vectors alongside frames so render workers can
        # reconstruct sample() without a fetch.
        if self._x_vec is not None:
            meta_path = self._meta_cache_path()
            np.savez_compressed(
                meta_path,
                x_vec=self._x_vec,
                y_vec=self._y_vec,
                crop_indices=np.array([
                    self._crop_row_start, self._crop_row_end,
                    self._crop_col_start, self._crop_col_end,
                ]),
                full_grid=np.array([self._full_grid_height, self._full_grid_width]),
                crop_computed=np.array([1 if self._crop_computed else 0]),
                bbox=np.array(self._bbox) if self._bbox else np.array([]),
            )

    def _read_cache(self, unix_ts: int) -> np.ndarray | None:
        path = self._cache_path_for(unix_ts)
        if not path.exists():
            return None
        try:
            return np.memmap(
                path, dtype=np.uint8, mode="r",
                shape=(self._grid_height, self._grid_width),
            )
        except Exception:
            logger.warning(
                "%s: failed to memmap %s, removing", self.friendly_name, path,
            )
            path.unlink(missing_ok=True)
            return None

    def _load_grid_meta(self) -> bool:
        """Load grid vectors + crop state from disk cache.

        Validates the cached BBOX against the current config — if the
        operator changed the BBOX, the cache is invalidated.
        """
        meta_path = self._meta_cache_path()
        if not meta_path.exists():
            return False
        try:
            data = np.load(meta_path, allow_pickle=False)
            cached_bbox = tuple(data["bbox"]) if len(data["bbox"]) == 4 else None
            current_bbox = self._bbox
            if cached_bbox != current_bbox:
                logger.info(
                    "%s: BBOX changed (%s → %s), invalidating cache",
                    self.friendly_name, cached_bbox, current_bbox,
                )
                meta_path.unlink(missing_ok=True)
                return False

            self._x_vec = data["x_vec"]
            self._y_vec = data["y_vec"]
            self._grid_width = len(self._x_vec)
            self._grid_height = len(self._y_vec)

            if "crop_indices" in data and int(data["crop_computed"][0]):
                ci = data["crop_indices"]
                self._crop_row_start = int(ci[0])
                self._crop_row_end = int(ci[1])
                self._crop_col_start = int(ci[2])
                self._crop_col_end = int(ci[3])
                self._crop_computed = True
            if "full_grid" in data:
                fg = data["full_grid"]
                self._full_grid_height = int(fg[0])
                self._full_grid_width = int(fg[1])

            return True
        except Exception:
            logger.warning("%s: failed to load grid meta", self.friendly_name)
            return False

    def _load_cached_frames(self) -> None:
        assert self._channel_cache_dir is not None
        if not self._load_grid_meta():
            return
        for entry in self._channel_cache_dir.glob("frame_*.dat"):
            try:
                unix_ts = int(entry.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)

    # ── Lifecycle ──

    async def close(self) -> None:
        return None

    # ── Cross-process snapshot (pickle for multi-worker) ──

    def __getstate__(self) -> dict:
        return {
            "cache_root": str(self._cache_root) if self._cache_root else None,
            "channel": self.channel,
            "timestamps": list(self._sorted_timestamps),
            "max_frames": self._max_frames,
            "bucket": self.s3_bucket,
            "bbox": self._bbox,
            "downsample_factor": self._downsample_factor,
        }

    def __setstate__(self, state: dict) -> None:
        cache_root = state.get("cache_root")
        self._cache_root = Path(cache_root) if cache_root else None
        self._max_frames = state.get("max_frames", 36)
        self._bbox = state.get("bbox")
        self._downsample_factor = state.get("downsample_factor", 1)
        self._frames = {}
        self._sorted_timestamps = []
        self._frame_keys = {}
        self._replaced_timestamps = []
        self._fs = None
        self._x_vec = None
        self._y_vec = None
        self._grid_height = 0
        self._grid_width = 0
        self._full_grid_height = 0
        self._full_grid_width = 0
        self._crop_row_start = 0
        self._crop_row_end = 0
        self._crop_col_start = 0
        self._crop_col_end = 0
        self._crop_computed = False
        self.name = self.friendly_name

        if self._cache_root is None:
            self._channel_cache_dir = None
            return
        self._channel_cache_dir = (
            self._cache_root / self._cache_subdir() / self.channel
        )
        if not self._channel_cache_dir.exists():
            return
        if not self._load_grid_meta():
            return
        for unix_ts in state.get("timestamps", []):
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)
