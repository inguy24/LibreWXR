# Adding a New Source

This guide walks through adding a radar composite, regional NWP grid, or satellite source to LibreWXR. By the end, your source will be auto-discovered at startup, dispatched to by `RadarFetcher` (radar), blended into the NWP chain (NWP), or wired into the satellite renderer (satellite), and need zero edits to `data/fetcher.py`, `data/regions.py`, or `data/coverage.py`.

If you're looking for the short procedural checklist instead of the full walkthrough, see the **"Adding a New Source"** section in [`CLAUDE.md`](../CLAUDE.md) at the project root.

Self-hosters running their own LibreWXR instance are free to integrate any source the architecture supports — that's between the operator and the upstream data provider. **Contributing a source to the core project is narrower:** the data has to be open, either by explicit license or by a clear governmental policy that places it in the open. Read [Upstream contribution criteria](#upstream-contribution-criteria) before doing the implementation work, especially if the licensing situation isn't already obvious.

## Table of Contents

- [Upstream contribution criteria](#upstream-contribution-criteria)
  - [Hard requirements](#hard-requirements)
  - [Soft constraints (Tier 2)](#soft-constraints-tier-2)
  - [Blockers](#blockers)
  - [When the license is unclear](#when-the-license-is-unclear)
- [How discovery works](#how-discovery-works)
- [Directory layout](#directory-layout)
  - [Country-dir convention](#country-dir-convention)
- [Adding a radar source](#adding-a-radar-source)
  - [`regions.py`](#regionspy)
  - [`stations.py`](#stationspy)
  - [`source.py`](#sourcepy)
  - [`__init__.py` (the provider)](#__init__py-the-provider)
- [Adding a regional NWP grid](#adding-a-regional-nwp-grid)
  - [`grid.py`](#gridpy)
  - [`__init__.py` (the provider)](#__init__py-the-provider-1)
  - [Priority numbers](#priority-numbers)
- [Config additions](#config-additions)
- [Coverage map](#coverage-map)
- [Tests](#tests)
- [Final checklist](#final-checklist)
- [Adding a satellite source](#adding-a-satellite-source)
  - [Directory layout (satellite)](#directory-layout-1)
  - [`source.py` (satellite)](#sourcepy-1)
  - [`__init__.py` (the provider, satellite)](#__init__py-the-provider-2)
  - [`SatelliteContribution` fields](#satellitecontribution-fields)
  - [Priority ordering (satellite)](#priority-ordering)
  - [BBOX crop](#bbox-crop)
  - [Auto-selection by operator location](#auto-selection-by-operator-location)
  - [Config additions (satellite)](#config-additions-1)
  - [Renderer compatibility](#renderer-compatibility)
  - [Final checklist (satellite-specific)](#final-checklist-satellite-specific)

## Upstream contribution criteria

LibreWXR is permissive about what you run on your own instance. Submitting a source for inclusion in the upstream project is narrower — it has to be redistributable by an AGPL-3.0 open-source project that anyone can host.

### Hard requirements

A source can be merged upstream when **all three** of the following hold:

- **Open license, or a site-wide open-data policy that plausibly covers it.** An explicit license is the easiest case — e.g., CC-BY-4.0 (MMD Malaysia), Etalab Open Licence v2.0 (Météo-France), OGDL v1.0 (CWA Taiwan), Canada OGL, US public domain. "Heavily implied" also counts when the operator publishes a clear blanket policy: e.g., a national meteorology site stating that unless marked otherwise, all data and information on the site is in the public domain. PAGASA (Philippines) is one example of that shape — site-wide public-domain notice covering products that lack their own per-product license text.
- **Anonymous access.** No API keys, no per-user tokens, no registration. API-key-gated sources are an automatic decline for the core project regardless of how generous the license is, because they don't scale to a self-hosted project where every operator would need to register their own credentials.
- **Redistribution permitted.** No "non-commercial only" clauses, no "no third-party hosting" clauses, no audience restrictions. LibreWXR does not put a ceiling on who can run it, and bundling sources whose terms exclude commercial or large-scale operators would create asymmetric licensing across the source set.

### Soft constraints (Tier 2)

These are accepted but worth flagging in the package `README.md`:

- WAFs / Cloudflare / Imperva in front of the source — tolerated when anonymous access works reliably in practice.
- Undocumented bounds or projection — workable when reverse-engineering is unambiguous and verifiable against the operator's own viewer at the same wall-clock moment.
- Attribution requirements — fine as long as they can be honored in the package `README.md` (and, if required by the license, in the served UI).
- Short archive depth (under 24 hours) — acceptable; note it in the README.

### Blockers

These rule a source out for upstream regardless of how technically attractive it is:

- API keys or per-user authentication.
- "Non-commercial use only" / "no redistribution" / "personal use only" terms.
- App-only access policies (operator's stated position is that the data is for their own apps, not third-party use).
- Paywalls, even nominal ones.
- Commercial state-owned-enterprise licensing (national met service that operates as a commercial entity and licenses data per customer).

### When the license is unclear

Reach out to the operator before doing the implementation work. Ask plainly whether the data is available for redistribution by a self-hostable, AGPL-3.0 open-source project. If the answer is no or ambiguous, record the determination in [`source-survey.md`](source-survey.md) — that file is the running ledger of accept / defer / reject decisions across radar and NWP sources, and is the right place to save the next contributor from re-discovering a blocker.

## How discovery works

At startup, `librewxr.sources.__init__` calls `pkgutil.walk_packages()` over the `sources/` tree, imports every subpackage, and collects two kinds of provider functions defined at the package level:

- `radar_provider(settings) -> RadarSourceContribution | None`
- `nwp_provider(settings, cache_dir) -> NWPContribution | None`

A provider returns `None` when its config flag is off — that's how a source opts out cleanly. The fetcher loops over `RADAR_PROVIDERS` and `NWP_PROVIDERS`, calls each one, and wires up only the contributions that returned non-`None`.

Side-effect: **adding a directory under `sources/` with a `radar_provider` or `nwp_provider` in its `__init__.py` is the entire registration step.** No central list to update.

## Directory layout

Every source is a self-contained Python package:

```
sources/
  world/                              # global sources
    ifs/
      __init__.py                     # provider
      grid.py                         # implementation
      interpolation.py                # source-specific helpers
      README.md                       # operator notes
  regional/
    <continent>/
      <country>/                      # only when the data is dominantly one country's
        radar/<source_name>/
          __init__.py
          source.py
          regions.py
          stations.py
          README.md
        nwp/<source_name>/
          __init__.py
          grid.py
          README.md
```

`__init__.py` is the only mandatory file the discovery walker looks at; everything else is convention so contributors land in predictable places.

### Country-dir convention

Use a country directory when the data is dominantly that country's (MRMS for the US, MMD for Malaysia, HRDPS for Canada). **Skip** the country directory when the data is genuinely multi-country (OPERA covers 30+ European countries; AROME Antilles covers Guadeloupe + Martinique + French Guiana; WRF-SMN covers six countries in the Southern Cone).

| Source       | Has country dir? | Why                                          |
|--------------|------------------|----------------------------------------------|
| MRMS (US)    | yes (`usa/`)     | NEXRAD-only network, single operator         |
| OPERA        | no               | EUMETNET, ~30 countries pooled               |
| HRDPS        | yes (`canada/`)  | ECCC's national model                        |
| AROME Antilles | no             | Multi-territory                              |
| WRF-SMN      | no               | AR+CL+UY+PY+BO+S.BR                          |
| MMD          | yes (`malaysia/`)| MET Malaysia, single operator                |
| ICON-EU      | no               | DWD-operated but EU-wide footprint           |

## Adding a radar source

Walk through with a hypothetical "ZQR Zorbia Composite" (single-country, single-region, ~150 km coverage).

### `regions.py`

Define one or more `RegionDef`s and expose them as a list plus a `REGION_GROUP` string. The discovery walker reads these from the package's `__init__.py` to populate the global `REGIONS` dict and `REGION_GROUPS` map.

```python
# sources/regional/.../zorbia/radar/zqr/regions.py
from librewxr.data.regions import RegionDef

ZQRCOMP = RegionDef(
    name="ZQRCOMP",
    west=12.0, east=22.0,
    south=-5.0, north=5.0,
    pixel_size=0.01,
    group="ZORBIA",
)

REGIONS = [ZQRCOMP]
```

`RegionDef` supports `proj="latlon"` (default), `proj="lcc"`, `proj="stere"`, and `proj="laea"`. When you need a projected grid, populate the relevant `grid_*` / `laea_*` / projection fields — see `OPERA` in `sources/regional/europe/radar/opera/regions.py` for an LAEA worked example.

### `stations.py`

This drives the coverage-mask builder. Two exports:

- `STATION_MAP: dict[str, list[tuple[float, float]]]` — one entry per region, each value a list of `(lat, lon)` for the contributing radars.
- `RANGE_OVERRIDES: dict[str, float]` — optional. Effective km range for each region. Regions missing here use the 240 km default Doppler reach.

```python
# sources/regional/.../zorbia/radar/zqr/stations.py
STATIONS = [
    (0.5,  15.2),    # ZQR-01 Lagos Central
    (-2.1, 18.7),    # ZQR-02 South Plateau
]

STATION_MAP = {"ZQRCOMP": STATIONS}
RANGE_OVERRIDES = {"ZQRCOMP": 150.0}  # operator publishes 150 km product
```

A range override is justified when the upstream publishes a non-standard footprint (OPERA's C-band at 300 km, El Salvador's 120 km product, CWA Taiwan's 450 km typhoon buffer, MET Malaysia's 350-375 km CAPPI). Drop the entry if 240 km is correct — don't repeat the default.

### `source.py`

The source class implements an async `fetch_frame(region) -> (datetime, np.ndarray) | None`, an `fetch_archive_frame(region, when)`, and a `close()`. The returned ndarray is `uint8` dBZ-encoded — use the canonical encoder in `librewxr.sources._helpers`:

```python
# sources/regional/.../zorbia/radar/zqr/source.py
from librewxr.sources._helpers import _dbz_float_to_uint8

class ZQRSource:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url)

    async def fetch_frame(self, region):
        ...  # fetch + decode native format → float32 dBZ
        return timestamp, _dbz_float_to_uint8(dbz_floats)

    async def fetch_archive_frame(self, region, when):
        ...

    async def close(self):
        await self._client.aclose()
```

If the upstream is GRIB2-based, wrap the cfgrib call in `_suppress_eccodes_stderr()` from `_helpers` to muzzle the library's non-actionable `dataTime` warnings.

### `__init__.py` (the provider)

This file is the public face of the package. It exports `REGIONS`, `REGION_GROUP`, the source class, and the `radar_provider` function the discovery walker calls.

```python
# sources/regional/.../zorbia/radar/zqr/__init__.py
from librewxr.sources._base import RadarSourceContribution

from .regions import REGIONS, ZQRCOMP
from .source import ZQRSource
from .stations import RANGE_OVERRIDES, STATION_MAP

REGION_GROUP = "ZORBIA"

__all__ = ["REGIONS", "REGION_GROUP", "ZQRCOMP", "ZQRSource", "radar_provider"]


def radar_provider(settings) -> RadarSourceContribution | None:
    if not getattr(settings, "zqr_enabled", True):
        return None
    instance = ZQRSource(settings.zqr_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )
```

A few things to note:

- The `getattr(settings, "zqr_enabled", True)` pattern lets the discovery walker run even if config setting isn't defined yet (e.g. during a partial migration). It also encodes the project convention: **new sources default to enabled** (see `feedback_models_enabled_by_default` in your memory if you have one).
- `RadarSourceContribution.preempts` exists for cross-source policy hints, currently unused — MRMS-vs-IEM dispatch lives in `data/fetcher.py` rather than being expressed declaratively. If you're adding a source that contests a region with an existing source, talk to the maintainer.
- The `regions=` field is the list of `RegionDef` objects the source can fetch frames for. The fetcher only wires the source to regions the user actually enabled via `LIBREWXR_ENABLED_REGIONS`.

## Adding a regional NWP grid

NWP sources are simpler — single file (`grid.py`) plus the provider in `__init__.py`. No regions, no stations.

### `grid.py`

Your `*Grid` class must satisfy the `NWPSource` Protocol in `data/nwp_source.py`:

- `name: str` (used in logs)
- `sample(lat, lon, timestamp, bilinear) -> uint8 dBZ-encoded ndarray`
- `supports_snow: bool` and `get_snow_mask(lat, lon, timestamp) -> bool ndarray`
- `domain_mask(lat, lon) -> bool ndarray` — where the grid has coverage
- `feather_mask(lat, lon) -> float32 ndarray in [0, 1]` — soft taper at the boundary
- `has_data_at(timestamp) -> bool` / `has_data() -> bool`
- `async close() -> None`

For the shared `compute_snow_mask` helper (used by every regional NWP that supports snow), import from HRRR:

```python
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import compute_snow_mask
```

HRRR was the original implementation so it owns the helper. DINI, ICON-EU, WRF-SMN, HRRR-Alaska, and JMA MSM all import it from this same location.

Interpolation helpers: if your source publishes at a coarser cadence than LibreWXR's 10-min stored interval, use the shared optical-flow interpolator in `data/nwp_interpolation.py`:

```python
from librewxr.data.nwp_interpolation import interpolate_run
```

IFS is the exception — its interpolation lives next door in `sources/world/ifs/interpolation.py` because it was the first one written and the design is IFS-shaped. The shared `nwp_interpolation` module came later when DINI / ICON-EU / WRF-SMN / JMA MSM needed it.

### `__init__.py` (the provider)

```python
# sources/regional/.../zorbia/nwp/zqr_wrf/__init__.py
from librewxr.sources._base import NWPContribution

from .grid import ZQRWRFGrid

__all__ = ["ZQRWRFGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    if not getattr(settings, "zqr_wrf_enabled", False):
        return None
    return NWPContribution(
        instance=ZQRWRFGrid(cache_dir=cache_dir),
        priority=35,
        name="ZQR-WRF",
    )
```

The `name` doubles as the source's identity in `/health` and in the
cross-process snapshot — it's auto-slugged to a key like
`zqr_wrf_grid` (lowercase, non-word chars → `_`, plus `_grid`). If
your display name doesn't slug cleanly (non-ASCII characters,
abbreviations like "NCaled" for "Nouvelle-Calédonie"), pass an
explicit `slug="…_grid"` field so the snapshot key stays stable:

```python
return NWPContribution(
    instance=ZQRWRFGrid(cache_dir=cache_dir),
    priority=35,
    name="ZQR-WRF Réunion",       # non-ASCII; auto-slug would mangle the é
    slug="zqr_wrf_reunion_grid",  # explicit key for /health + state.json
)
```

The fetcher dispatch in `data/fetcher.py` calls every grid's `fetch()`
under a shared semaphore. It uses `inspect.signature` to decide whether
to pass `history_seconds` and `horizon_seconds` — if your grid's
`fetch()` accepts those kwargs (the common case), they're passed; if
it takes no kwargs (like IFS), nothing is passed. No registration step
is needed for either shape.

### Priority numbers

`priority` controls position in the `NWPChain`: **lower numbers run first**, so narrower / higher-resolution domains should pick smaller numbers. The chain dispatcher fills pixels from the first source whose `feather_mask` is positive, falling through to the next source where the previous one's feather is zero.

Current assignments:

| Source                   | Priority | Notes                                              |
|--------------------------|----------|----------------------------------------------------|
| HRRR                     | 10       | 3 km CONUS, narrowest domain                       |
| HRRR-Alaska              | 11       | Same model, disjoint domain                        |
| HRDPS                    | 20       | 2.5 km Canada                                      |
| JMA MSM                  | 20       | 5 km Japan + Korean Peninsula + Taiwan (disjoint from HRDPS) |
| AROME Antilles           | 25       | 2.5 km Caribbean (FR-GP + MQ)                      |
| AROME Guyane             | 26       | 2.5 km French Guiana                               |
| AROME Indien             | 27       | 2.5 km SW Indian Ocean (RE + YT + KM + MG + SW)    |
| AROME Nouvelle-Calédonie | 28       | 2.5 km SW Pacific (NC + VU)                        |
| AROME Polynésie          | 29       | 2.5 km French Polynesia                            |
| DMI DINI                 | 30       | 2 km Nordic / NW Europe                            |
| ICON-EU                  | 35       | 7 km Europe (catches what DMI DINI doesn't)        |
| WRF-SMN                  | 40       | 4 km Southern Cone                                 |
| **IFS**                  | **1000** | Global catch-all                                   |

Pick a number that places your source in the right spot. If your source's domain is disjoint from every other regional, the exact number between 10 and 999 doesn't matter behaviorally — just keep it deterministic and self-documenting. If your source overlaps another regional, put it before or after based on which should win inside the overlap.

## Config additions

The Pydantic `Settings` class in `src/librewxr/config.py` is the single source of truth for all `LIBREWXR_*` env vars. Add your source's settings alongside the existing ones:

```python
# config.py
class Settings(BaseSettings):
    ...
    zqr_enabled: bool = True             # default-enabled per project convention
    zqr_base_url: str = "https://radar.zorbia.example/composite/"
```

Then add the same line to `.env.example` with a brief comment so users discover the knob. The full reference (`docs/configuration-reference.md`) is where you'd document the setting in detail.

## Coverage map

Add the new station footprint to `scripts/generate_coverage_map.py`:

```python
# in build_radar_sources():
for poly in union_of_radar_circles(ZQR_STATIONS, range_for("ZQRCOMP")):
    radar.append(Source("ZQR Composite (Zorbia)", "#ff7f0e", poly))
```

Import the station list and any range override directly from your new `stations.py`:

```python
from librewxr.sources.regional.<...>.radar.zqr.stations import (
    RANGE_OVERRIDES as ZQR_RANGES,
    STATIONS as ZQR_STATIONS,
)
# then merge ZQR_RANGES into REGION_RADAR_RANGE at the top of the script
```

Regenerate the PNGs (the script header documents the throwaway-venv recipe):

```bash
python scripts/generate_coverage_map.py
```

Commit both updated PNGs (`docs/coverage-map-radar.png` and `docs/coverage-map-models.png`) alongside your code change.

## Tests

Add a `tests/test_<source>.py` for your source. Mark it appropriately:

```python
import pytest
pytestmark = pytest.mark.sources  # or pytest.mark.nwp for grids
```

Look at `tests/test_cwa.py` (smallest radar test) or `tests/test_arome_antilles_grid.py` (smallest NWP test) for the smallest viable shape. Mock HTTP with `httpx.MockTransport` or the project's existing fixtures — don't hit the network in unit tests.

If the registry-discovery tests in `tests/test_sources_discovery.py` need an assertion for your new source (e.g. that it shows up in the chain or in `RADAR_PROVIDERS`), add it there.

## Final checklist

Before opening a PR:

- [ ] License + redistribution terms verified compatible with [Upstream contribution criteria](#upstream-contribution-criteria).
- [ ] Package directory exists under the right `sources/.../` path (follow the country-dir convention).
- [ ] `__init__.py` exports a `radar_provider` or `nwp_provider`.
- [ ] `regions.py` (radar) defines `REGIONS` and `REGION_GROUP`.
- [ ] `stations.py` (radar) defines `STATION_MAP` and (optionally) `RANGE_OVERRIDES`.
- [ ] New config knobs added to `config.py` and `.env.example`, defaulting to enabled.
- [ ] License + attribution documented in the package's `README.md` (every existing source has one — copy the shape).
- [ ] `scripts/generate_coverage_map.py` updated; both coverage PNGs regenerated.
- [ ] `pytest` clean.
- [ ] Smoke-run the server (`python -m librewxr.main`) once and confirm the new source shows up in the startup logs.

That's it — no `data/fetcher.py`, `data/regions.py`, or `data/coverage.py` edits required. The discovery walker handles registration and the providers do the rest.

## Adding a satellite source

Satellite sources live under `sources/satellite/<name>/` and follow the same auto-discovery pattern as radar and NWP: the discovery walker in `sources/__init__.py` imports every package under `sources/satellite/`, calls its `satellite_provider(settings, cache_dir)` function, and flattens the returned list of `SatelliteContribution` objects into the global satellite registry.

### Directory layout

```
sources/
  satellite/
    <name>/
      __init__.py     # exports satellite_provider()
      source.py       # SatelliteSource implementation(s)
      README.md       # operator notes, licensing, attribution
```

No `regions.py` or `stations.py` — satellite sources don't feed the radar coverage-mask builder.

### `source.py`

Your source class must satisfy the `SatelliteSource` Protocol in `sources/_base.py`:

- `name: str` — display name (e.g. `"GOES-18 IR"`)
- `timestamps: list[int]` — sorted Unix-epoch list of loaded frames
- `loaded: bool` — `True` when at least one frame is available
- `data_bytes: int` — total memory across all frames (surfaced via `/health`)
- `async fetch() -> bool` — ingest new frames from upstream; return `True` if new data arrived
- `sample(lat: ndarray, lon: ndarray, timestamp: int | None) -> ndarray` — nearest-neighbour lookup into the stored grid; returns `uint8` array shaped like `lat`/`lon`; **0 = no data** (pixel not visible to the sensor, outside grid bounds, or no frame loaded)
- `async close() -> None` — cleanup
- `__getstate__` / `__setstate__` — pickle support for multi-worker snapshot (same memmap-reload pattern as GMGSI)

The `sample()` contract:

- **Inputs:** `lat` and `lon` are `float64` ndarrays of identical shape (tile pixel grids from the renderer).
- **Output:** `uint8` ndarray of the same shape. Values are sensor-specific encoded brightness (IR channels map brightness temperature to 0–255; VIS channels map reflectance factor to 0–255). `0` means no data — the renderer treats it as transparent.
- **Projection:** the source handles its own projection internally. For equirectangular grids (GMGSI), this is direct array indexing. For geostationary sources (GOES, Himawari), it's a forward lat/lon → scan-angle transform followed by nearest-neighbour grid lookup (see `tiles/geostationary.py` and `sources/satellite/_geo_base.py`).

For geostationary satellites, inherit from `GeoSatSource` in `sources/satellite/_geo_base.py`. It handles S3 listing, frame retention, geostationary projection, BBOX crop, disk caching, and multi-worker serialization. Subclasses pin satellite-specific class variables (`sat_lon`, `sat_height`, `s3_bucket`, `s3_product_path`, `s3_filename_token`, `friendly_name`, `channel`, `cadence_minutes`) and override `_decode_netcdf()` for sensor-specific value mapping.

### `__init__.py` (the provider)

The package-level `satellite_provider(settings, cache_dir)` returns a list of `SatelliteContribution` objects — one per channel (e.g. IR and VIS). Return `[]` when the source doesn't apply (wrong region, disabled by config).

```python
# sources/satellite/<name>/__init__.py
from librewxr.sources._base import SatelliteContribution

from .source import MyIRSource, MyVISSource

__all__ = ["MyIRSource", "MyVISSource", "satellite_provider"]


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    if not getattr(settings, "my_sat_enabled", True):
        return []

    # Check operator location — return [] when outside coverage
    center_lon = _center_longitude(settings)
    if center_lon is None or not _in_coverage(center_lon):
        return []

    retention = getattr(settings, "satellite_max_frames", 36)
    bbox = getattr(settings, "get_bbox", lambda: None)()
    contributions: list[SatelliteContribution] = []

    if getattr(settings, "my_sat_ir_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=MyIRSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                ),
                priority=5,
                name="MySat IR",
                slug="mysat_ir_grid",
            ),
        )

    if getattr(settings, "my_sat_vis_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=MyVISSource(
                    cache_dir=cache_dir, max_frames=retention, bbox=bbox,
                ),
                priority=6,
                name="MySat VIS",
                slug="mysat_vis_grid",
            ),
        )

    return contributions
```

### `SatelliteContribution` fields

| Field | Type | Description |
|-------|------|-------------|
| `instance` | `SatelliteSource` | The source object (IR or VIS channel) |
| `priority` | `int` | Lower number = higher priority. When multiple sources cover the same pixel, the lower-priority source wins |
| `name` | `str` | Display name for logs and `/health` |
| `slug` | `str \| None` | Explicit key for the `/health` and state.json snapshot. Leave `None` to auto-derive from `name` |

### Priority ordering

Priority controls which source wins when multiple satellite providers return contributions for the same deployment. The renderer walks contributions by priority (lower first) and uses the first source that has data for a given pixel.

Current assignments:

| Source | IR Priority | VIS Priority | Notes |
|--------|------------|-------------|-------|
| GOES-18/19 | 5 | 6 | High-res regional, Americas |
| Himawari-9 | 5 | 6 | High-res regional, Asia-Pacific |
| GMGSI | 10 | 11 | Global fallback |

GOES and Himawari share priority 5/6 because their coverage zones don't overlap — auto-selection ensures only one returns contributions for a given station. Pick a priority below 10 for any new high-resolution regional source; GMGSI at 10/11 is the global catch-all.

### BBOX crop

Sources should accept a `bbox` parameter (a `(north, west, south, east)` tuple or `None`) and crop their stored grids to reduce memory. The `GeoSatSource` base class handles this automatically for geostationary sources — it converts the BBOX corners to scan-angle space, computes pixel-index bounds, and slices every frame after decode. For non-geostationary sources (equirectangular grids like GMGSI), implement cropping in the source's `_decode` path.

### Auto-selection by operator location

Each satellite provider should check the operator's location and return `[]` when outside its coverage zone. The standard pattern reads the BBOX center or `LIBREWXR_STATION_LON`:

```python
def _center_longitude(settings) -> float | None:
    bbox = getattr(settings, "get_bbox", lambda: None)()
    if bbox is not None:
        _, west, _, east = bbox
        return (west + east) / 2.0

    station_lon = getattr(settings, "station_lon", None)
    if station_lon is not None:
        return float(station_lon)

    return None
```

This way, when no regional source covers the station, all regional providers return `[]` and GMGSI takes over as the global fallback. The discovery walker handles this gracefully — an empty list is the same as "not applicable."

### Config additions

Add your source's toggle to `config.py` alongside the existing satellite settings, and mirror it in `.env.example`:

```python
# config.py
class Settings(BaseSettings):
    ...
    my_sat_enabled: bool = True
    my_sat_ir_enabled: bool = True
    my_sat_vis_enabled: bool = True
```

### Renderer compatibility

The satellite tile renderer in `tiles/satellite_renderer.py` is source-agnostic — it calls `source.sample(lat, lon, timestamp)` and composites the VIS-over-IR result. No per-source changes to the renderer are needed as long as:

1. IR channels map brightness temperature to uint8 using a consistent scale (cold cloud tops = high values, warm ground = low values, matching GMGSI's encoding).
2. VIS channels map reflectance to uint8 (0 = dark/night, 255 = bright).
3. `sample()` returns 0 for no-data pixels.

If your source's native value range differs from GMGSI's encoding (e.g. GOES CMI returns Kelvin float32), map to uint8 in `_decode_netcdf()` before storing the frame.

### Final checklist (satellite-specific)

- [ ] Package directory exists under `sources/satellite/<name>/`.
- [ ] `__init__.py` exports a `satellite_provider(settings, cache_dir) -> list[SatelliteContribution]`.
- [ ] Source class satisfies the `SatelliteSource` Protocol (`name`, `timestamps`, `loaded`, `data_bytes`, `fetch()`, `sample()`, `close()`, `__getstate__`/`__setstate__`).
- [ ] `sample()` returns uint8, 0 = no data, same shape as input lat/lon arrays.
- [ ] Auto-selection returns `[]` when the station is outside coverage.
- [ ] BBOX crop is supported (accepts `bbox` parameter, crops stored grids).
- [ ] Config toggles added to `config.py` and `.env.example`.
- [ ] License + attribution documented in the package's `README.md`.
- [ ] `pytest` clean.
- [ ] Smoke-run confirms the source appears in startup logs and `/health`.
