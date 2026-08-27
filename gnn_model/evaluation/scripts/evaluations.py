#!/usr/bin/env python
"""evaluations.py

Entry point for OCELOT evaluation plotting and metrics.

Originally authored by Azadeh Gholoubi. Restructured so that:

  * configuration lives in plotting.yaml rather than in the sbatch script
  * the instrument table is a registry, not a 500-line if-chain
  * every CSV is discovered once and read once, then all requested figures
    are rendered from the in-memory frame
  * ground truth is detected from the file rather than declared by the caller
  * AR rollout steps (obs-space-ar<N>) are first-class selectors

Usage
-----
    python evaluations.py --config plotting.yaml
    python evaluations.py --config plotting.yaml --filedetection
    python evaluations.py --config plotting.yaml --instruments atms --max-files 2
    python evaluations.py --config plotting.yaml --mode metrics
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "figures"))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "plotting.yaml")


# =============================================================================
# Instrument registry
# =============================================================================

@dataclass(frozen=True)
class InstrumentSpec:
    """Per-instrument plotting behaviour.

    Adding an instrument is one entry here instead of six call sites.
    """
    n_channels: int
    units: str | None = None
    error_metric: str = "percent"          # auto | absolute | percent | smape
    figures: tuple[str, ...] = ("diff", "error_map")
    drop_small_truth: bool = False
    profile_min_samples: int | None = None
    channels: tuple = ()                   # filled in from config at runtime


# Populated from plotting.yaml by build_registry(). Empty until a config is
# loaded -- there is no hardcoded copy of the instrument table in this file.
REGISTRY: dict[str, InstrumentSpec] = {}

_SPEC_FIELDS = {
    "n_channels": int,
    "units": (str, type(None)),
    "error_metric": str,
    "figures": tuple,
    "drop_small_truth": bool,
    "profile_min_samples": (int, type(None)),
}

VALID_ERROR_METRICS = {"auto", "absolute", "percent", "smape"}


def build_registry(cfg: dict) -> dict[str, InstrumentSpec]:
    """Turn instruments.registry in the YAML into InstrumentSpec objects.

    Entries inherit from instruments.defaults, so most are one line.
    """
    node = cfg.get("instruments") or {}
    raw = node.get("registry")
    if not raw:
        raise ValueError(
            "plotting.yaml is missing instruments.registry. This is where the "
            "instrument table lives; see the shipped plotting.yaml for the shape."
        )
    base = dict(node.get("defaults") or {})

    out: dict[str, InstrumentSpec] = {}
    for name, entry in raw.items():
        entry = {**base, **(entry or {})}
        unknown = set(entry) - set(_SPEC_FIELDS)
        if unknown:
            raise ValueError(
                f"instruments.registry.{name}: unknown key(s) "
                f"{sorted(unknown)}; allowed: {sorted(_SPEC_FIELDS)}"
            )
        if "n_channels" not in entry:
            raise ValueError(f"instruments.registry.{name}: n_channels is required")

        metric = str(entry.get("error_metric", "percent"))
        if metric not in VALID_ERROR_METRICS:
            raise ValueError(
                f"instruments.registry.{name}: error_metric={metric!r} is not one "
                f"of {sorted(VALID_ERROR_METRICS)}"
            )

        figs = entry.get("figures") or ["diff", "error_map"]
        out[name] = InstrumentSpec(
            n_channels=int(entry["n_channels"]),
            units=entry.get("units"),
            error_metric=metric,
            figures=tuple(figs),
            drop_small_truth=bool(entry.get("drop_small_truth", False)),
            profile_min_samples=entry.get("profile_min_samples"),
        )

    REGISTRY.clear()
    REGISTRY.update(out)
    return out


# =============================================================================
# Config loading
# =============================================================================

# Required keys, as dotted paths. This is a schema -- key names and expected
# shapes only. It holds no values: plotting.yaml is the single source for those.
# shapes only. It holds no values: plotting.yaml is the single source for those.
SCHEMA: tuple[tuple[str, str], ...] = (
    ("io.base_dir",                    "path or null"),
    ("io.data_dir",                    "path or list of paths"),
    ("io.plot_dir",                    "path"),
    ("io.has_ground_truth",            "auto | true | false"),
    ("io.ar_dir_patterns",             "list of regexes, or null for defaults"),
    ("select.dates",                   "selector or null"),
    ("select.date_step_hours",         "int"),
    ("select.epochs",                  "selector, 'latest', or null"),
    ("select.batches",                 "selector or null"),
    ("select.fhrs",                    "selector or null"),
    ("select.ar_steps",                "selector or null"),
    ("select.max_files",               "int or null"),
    ("select.max_files_per_instrument", "int or null"),
    ("instruments.enabled",            "list of names or null"),
    ("instruments.channels",           "mapping name -> list or null"),
    ("instruments.registry",           "mapping name -> spec"),
    ("figures.diff",                   "bool"),
    ("figures.error_map",              "bool"),
    ("figures.profiles",               "bool"),
    ("figures.pressure_levels",        "bool"),
    ("figures.wind",                   "bool"),
    ("figures.pred_only",              "auto | true | false"),
    ("figures.ar_growth",              "bool"),
    ("figures.metrics_table",          "bool"),
    ("features.auto_absolute",         "list of feature names"),
    ("features.tiny_threshold",        "mapping feature -> number"),
    ("features.calm_wind_threshold",   "number, m/s"),
    ("features.qc_ranges",             "mapping feature -> [min, max]"),
    ("windows.subwindows",             "bool"),
    ("windows.subwindow_leads",        "auto or list of ints"),
    ("windows.horizons",               "bool"),
    ("windows.horizon_length_hours",   "int"),
    ("windows.horizon_bounds",         "auto or list of [lo, hi]"),
    ("windows.strict_obs_window",      "bool"),
    ("limits.mode",                    "auto | robust | fixed"),
    ("limits.robust_percentile",       "number"),
    ("limits.share_across_ar_steps",   "bool"),
    ("limits.fixed",                   "mapping, may be empty"),
    ("render.dpi",                     "int"),
    ("render.point_size",              "number"),
    ("render.cmap_value",              "colormap name"),
    ("render.cmap_diff",               "colormap name"),
    ("render.cmap_error",              "colormap name"),
    ("render.min_points_per_level",    "int"),
    ("render.profile_min_samples",     "int"),
    ("render.jobs",                    "int"),
)


def dotted_get(cfg: dict, path: str):
    """Fetch a dotted key. Raises KeyError naming the full path if absent."""
    node = cfg
    for i, part in enumerate(path.split(".")):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"Missing config key '{path}'. Add it to plotting.yaml "
                f"(run --print-config to see what is currently loaded)."
            )
        node = node[part]
    return node


def validate_config(cfg: dict, source: str) -> dict:
    """Fail loudly on a missing key rather than silently substituting a value.

    A silent fallback is how a plot quietly changes without anyone noticing,
    so an incomplete config is an error, not a warning.
    """
    missing = []
    for path, shape in SCHEMA:
        try:
            dotted_get(cfg, path)
        except KeyError:
            missing.append((path, shape))
    if missing:
        lines = "\n".join(f"    {p:<34} ({shape})" for p, shape in missing)
        raise ValueError(
            f"{source} is missing {len(missing)} required key(s):\n{lines}\n"
            f"  Copy them from the plotting.yaml shipped alongside this script."
        )

    if str(dotted_get(cfg, "limits.mode")).lower() not in ("auto", "robust", "fixed"):
        raise ValueError("limits.mode must be one of: auto, robust, fixed")
    return cfg


def load_config(path: str | None) -> dict:
    if not path:
        raise ValueError(
            "A config file is required. Pass --config plotting.yaml "
            "(there are no built-in defaults; plotting.yaml is the only "
            "source of settings)."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required to read plotting.yaml. "
            "conda install -n gnn-env pyyaml"
        ) from e
    class _StrictLoader(yaml.SafeLoader):
        """SafeLoader that rejects duplicate keys.

        Stock YAML silently keeps the last of a repeated key, so a copy-paste
        slip leaves two blocks in the file and only one of them in effect.
        """

    def _no_dupes(loader, node, deep=False):
        mapping = {}
        for k_node, v_node in node.value:
            k = loader.construct_object(k_node, deep=deep)
            if k in mapping:
                raise ValueError(
                    f"{path}: duplicate key '{k}' at line "
                    f"{k_node.start_mark.line + 1} (first seen earlier). "
                    f"YAML would silently keep only the last one."
                )
            mapping[k] = loader.construct_object(v_node, deep=deep)
        return mapping

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes)

    with open(path, "r") as fh:
        cfg = yaml.load(fh, Loader=_StrictLoader) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse to a mapping.")
    validate_config(cfg, path)
    build_registry(cfg)
    return cfg


def _under_base(path: str, base: str | None) -> str:
    """Join a possibly-relative path onto base_dir. Absolute paths win."""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or not base:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base, path))


def resolve_paths(cfg: dict) -> dict:
    """Expand io.base_dir into data_dir / plot_dir.

    Called once after CLI overrides so everything downstream sees absolute
    paths and no other function has to know base_dir exists.
    """
    io = cfg["io"]
    base = io.get("base_dir")
    base = os.path.abspath(os.path.expanduser(str(base))) if base else None
    io["base_dir"] = base

    if base and not os.path.isdir(base):
        print(f"[WARN] io.base_dir does not exist: {base}")

    dd = io.get("data_dir")
    io["data_dir"] = ([_under_base(p, base) for p in dd]
                      if isinstance(dd, (list, tuple))
                      else _under_base(dd, base))
    io["plot_dir"] = _under_base(io.get("plot_dir"), base)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.base_dir:
        cfg["io"]["base_dir"] = args.base_dir
    if args.data_dir:
        cfg["io"]["data_dir"] = args.data_dir
    if args.plot_dir:
        cfg["io"]["plot_dir"] = args.plot_dir
    if args.instruments:
        cfg["instruments"]["enabled"] = list(args.instruments)
    if args.max_files is not None:
        cfg["select"]["max_files"] = (None if args.max_files < 0 else args.max_files)
    if args.ar_steps:
        cfg["select"]["ar_steps"] = [int(x) for x in args.ar_steps]
    if args.dates:
        cfg["select"]["dates"] = list(args.dates)
    if args.epoch is not None:
        cfg["select"]["epochs"] = args.epoch
    if args.batch_idx is not None:
        cfg["select"]["batches"] = args.batch_idx
    if args.fhr is not None:
        cfg["select"]["fhrs"] = args.fhr
    if args.subwindow_leads:
        cfg["windows"]["subwindow_leads"] = [int(x) for x in args.subwindow_leads]
    if args.no_horizons:
        cfg["windows"]["horizons"] = False
    if args.strict_obs_window:
        cfg["windows"]["strict_obs_window"] = True
    if args.has_ground_truth is not None:
        cfg["io"]["has_ground_truth"] = args.has_ground_truth
    return cfg


# =============================================================================
# File discovery
# =============================================================================

FNAME_RE = re.compile(
    r"^(?:val|pred)_(?P<inst>.+?)"
    r"(?:_init_(?P<init>\d{10}))?"
    r"(?:_f(?P<fhr>\d{3}))?"
    r"(?:_epoch(?P<epoch>\d+))?"
    r"(?:_batch(?P<batch>\d+))?"
    r"(?:_step\d+)?\.csv$"
)

# mesh-grid artifacts: <instrument>_init_<YYYYMMDDHH>_f<FHR>[_epochE_batchB].csv
MESH_RE = re.compile(
    r"^(?P<inst>.+?)_init_(?P<init>\d{10})_f(?P<fhr>\d{3})"
    r"(?:_epoch(?P<epoch>\d+))?(?:_batch(?P<batch>\d+))?\.csv$"
)

# Default AR directory forms. Each pattern is fullmatched against a whole path
# component and must capture the step number. Because the match must cover the
# whole name, r"ar(\d+)" cannot match "obs-space-ar2", so order is irrelevant.
DEFAULT_AR_DIR_PATTERNS = (
    r"obs-space-ar(\d+)",
    r"mesh-grid-ar(\d+)",
    r"ar(\d+)",
)


def _ar_dir_regexes(cfg: dict | None):
    pats = ((cfg or {}).get("io") or {}).get("ar_dir_patterns")
    if not pats:
        pats = DEFAULT_AR_DIR_PATTERNS
    return [re.compile(p, re.IGNORECASE) for p in pats]


def _normalize_instrument(raw: str) -> tuple[str | None, str | None]:
    """Map a filename stem onto a registry key, tolerating suffixes.

    Prediction CSVs are written with variant suffixes such as
    'aircraft_target' or 'surface_obs_target'. Match the longest registry key
    that the stem starts with and keep the remainder as a variant label so
    variants do not overwrite each other's figures.
    """
    if raw in REGISTRY:
        return raw, None
    cands = [k for k in REGISTRY if raw.startswith(k + "_")]
    if cands:
        key = max(cands, key=len)
        return key, raw[len(key) + 1:]
    return None, None


@dataclass(frozen=True)
class FileRecord:
    path: str
    instrument: str
    rel_dir: str = ""          # source subdirectory, relative to its data_dir root
    variant: str | None = None
    init_time: str | None = None
    epoch: int | None = None
    batch: int | None = None
    fhr: int | None = None
    ar_step: int | None = None
    has_truth: bool | None = None

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


def _parse_filename(path: str) -> dict | None:
    base = os.path.basename(path)
    m = FNAME_RE.match(base) or MESH_RE.match(base)
    if not m:
        return None
    g = m.groupdict()
    return {
        "instrument": g.get("inst"),
        "init_time": g.get("init"),
        "epoch": int(g["epoch"]) if g.get("epoch") else None,
        "batch": int(g["batch"]) if g.get("batch") else None,
        "fhr": int(g["fhr"]) if g.get("fhr") else None,
    }


def _ar_step_from_path(path: str, regexes=None) -> int | None:
    """Parse an AR rollout step from any directory component of the path."""
    regexes = regexes if regexes is not None else _ar_dir_regexes(None)
    for part in os.path.normpath(path).split(os.sep):
        for rx in regexes:
            m = rx.fullmatch(part)
            if m:
                return int(m.group(1))
    return None


@dataclass(frozen=True)
class FileRecord:
    path: str
    instrument: str
    rel_dir: str = ""          # source subdirectory, relative to its data_dir root
    variant: str | None = None
    init_time: str | None = None
    epoch: int | None = None
    batch: int | None = None
    fhr: int | None = None
    ar_step: int | None = None
    has_truth: bool | None = None

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


def _parse_filename(path: str) -> dict | None:
    base = os.path.basename(path)
    m = FNAME_RE.match(base) or MESH_RE.match(base)
    if not m:
        return None
    g = m.groupdict()
    return {
        "instrument": g.get("inst"),
        "init_time": g.get("init"),
        "epoch": int(g["epoch"]) if g.get("epoch") else None,
        "batch": int(g["batch"]) if g.get("batch") else None,
        "fhr": int(g["fhr"]) if g.get("fhr") else None,
    }


def _ar_step_from_path(path: str, regexes=None) -> int | None:
    """Parse an AR rollout step from any directory component of the path."""
    regexes = regexes if regexes is not None else _ar_dir_regexes(None)
    for part in os.path.normpath(path).split(os.sep):
        for rx in regexes:
            m = rx.fullmatch(part)
            if m:
                return int(m.group(1))
    return None


def _ar_step_from_fhr(fhr: int | None, window_hours) -> int | None:
    """Infer the AR step from a forecast hour when no AR directory exists.

    val_mesh_csv writes a flat tree where the rollout is encoded only in the
    filename: with a 12h window, f003-f012 is AR0, f015-f024 is AR1, and so on.
    """
    if fhr is None or not window_hours:
        return None
    w = int(window_hours)
    if w <= 0 or int(fhr) <= 0:
        return None
    return (int(fhr) - 1) // w


def _detect_truth(path: str) -> bool:
    """Header-only probe for any true_* column."""
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return any(c.startswith("true_") for c in head.columns)


def scan(cfg: dict, unmatched: list | None = None) -> list[FileRecord]:
    roots = cfg["io"]["data_dir"]
    roots = [roots] if isinstance(roots, str) else list(roots)
    regexes = _ar_dir_regexes(cfg)

    paths: list[tuple[str, str]] = []   # (path, root)
    for root in roots:
        # Always recursive: AR steps live in subdirectories of data_dir.
        pattern = os.path.join(root, "**", "*.csv")
        paths.extend((p, root) for p in glob.glob(pattern, recursive=True))

    seen: set[str] = set()
    records: list[FileRecord] = []
    for p, root in sorted(paths):
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        if os.path.basename(p).endswith("_level_skill.csv"):
            continue  # our own output
        parsed = _parse_filename(p)
        if not parsed or not parsed["instrument"]:
            if unmatched is not None:
                unmatched.append((p, "filename pattern not recognised"))
            continue
        raw = parsed.pop("instrument")
        inst, variant = _normalize_instrument(raw)
        if inst is None:
            if unmatched is not None:
                unmatched.append((p, f"instrument '{raw}' not in REGISTRY"))
            continue
        ar = _ar_step_from_path(p, regexes)
        rel = os.path.relpath(os.path.dirname(p), root)
        records.append(FileRecord(
            path=p, instrument=inst, variant=variant,
            rel_dir="" if rel == "." else rel,
            ar_step=ar, **parsed
        ))

    # Sort numerically. Sorting by path string puts epoch11 before epoch12
    # because "11" < "120" < "12" lexicographically.
    def _key(r: FileRecord):
        big = 10 ** 9
        return (
            r.instrument, r.variant or "",
            r.init_time or "",
            r.ar_step if r.ar_step is not None else -1,
            r.fhr if r.fhr is not None else big,
            r.epoch if r.epoch is not None else big,
            r.batch if r.batch is not None else big,
            r.path,
        )

    records.sort(key=_key)
    return records


# ---------------------------------------------------------------------------
# Selector matching
# ---------------------------------------------------------------------------

_WARNED_SELECTORS: set[str] = set()


def _as_selector(spec, name: str) -> tuple[str, Any]:
    """Normalise a selector into ('any'|'list'|'range', payload).

    Grammar, deliberately unambiguous:
        null / "auto"        -> no filter
        12                   -> exactly 12
        [12, 120]            -> exactly 12 or exactly 120   (a LIST)
        {range: [12, 120]}   -> every value from 12 to 120  (a RANGE)
        {list: [12, 120]}    -> same as the bare list
        "latest"             -> handled separately, epochs only

    A bare two-element list used to be read as a range. It is now a list, so
    that `epochs: [12, 120]` ported from a shell array EPOCHS=(12 120) means
    those two epochs and not the 109 in between.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("auto", "all", "")):
        return "any", None
    if isinstance(spec, dict):
        if "range" in spec:
            lo, hi = spec["range"]
            return "range", (lo, hi)
        if "list" in spec:
            return "list", list(spec["list"])
        raise ValueError(f"select.{name}: expected 'range' or 'list' key, got {spec}")
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
        if len(vals) == 2 and name not in _WARNED_SELECTORS:
            _WARNED_SELECTORS.add(name)
            print(f"[select] {name}={vals} is read as two exact values. "
                  f"For an inclusive range write  {name}: {{range: {vals}}}")
        return "list", vals
    return "list", [spec]


def _expand_dates(spec, step_hours: int) -> set[str] | None:
    kind, payload = _as_selector(spec, "dates")
    if kind == "any":
        return None
    if kind == "list":
        return {str(x) for x in payload}

    lo, hi = (str(x) for x in payload)
    try:
        start = datetime.strptime(lo, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        end = datetime.strptime(hi, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(
            f"select.dates range endpoints must be YYYYMMDDHH, got {lo!r}, {hi!r}"
        ) from e

    out, cur = set(), start
    step = timedelta(hours=max(int(step_hours), 1))
    while cur <= end:
        out.add(cur.strftime("%Y%m%d%H"))
        cur += step
    return out


def _match_num(value, spec, name: str = "selector") -> bool:
    kind, payload = _as_selector(spec, name)
    if kind == "any":
        return True
    if value is None:
        return False
    if kind == "range":
        lo, hi = int(payload[0]), int(payload[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo <= int(value) <= hi
    return int(value) in {int(x) for x in payload}


def _keep_latest_epoch(records: list[FileRecord]) -> list[FileRecord]:
    """Keep only the highest epoch within each otherwise-identical group.

    Mesh dumps write one file per epoch, so a single init/fhr can have several
    hundred. `epochs: latest` collapses those to the newest checkpoint.
    """
    best: dict[tuple, FileRecord] = {}
    passthrough: list[FileRecord] = []
    for r in records:
        if r.epoch is None:
            passthrough.append(r)
            continue
        key = (r.instrument, r.variant, r.init_time, r.fhr, r.batch, r.ar_step)
        cur = best.get(key)
        if cur is None or r.epoch > cur.epoch:
            best[key] = r
    out = passthrough + list(best.values())
    if len(out) < len(records):
        print(f"[select] epochs=latest kept {len(out)} of {len(records)} file(s).")
    return out


def apply_filters(records: list[FileRecord], cfg: dict) -> list[FileRecord]:
    sel = cfg["select"]
    enabled = cfg["instruments"].get("enabled")
    enabled_set = {str(x).strip() for x in enabled} if enabled else None
    dates = _expand_dates(sel.get("dates"), int(sel.get("date_step_hours", 12) or 12))

    out = []
    for r in records:
        if enabled_set is not None and r.instrument not in enabled_set:
            continue
        if dates is not None and (r.init_time is None or r.init_time not in dates):
            continue
        if str(sel.get("epochs")).lower() != "latest" and \
                not _match_num(r.epoch, sel.get("epochs"), "epochs"):
            continue
        if not _match_num(r.batch, sel.get("batches"), "batches"):
            continue
        if sel.get("fhrs") not in (None, "auto") and not _match_num(r.fhr, sel.get("fhrs"), "fhrs"):
            continue
        if sel.get("ar_steps") not in (None, "auto") and not _match_num(r.ar_step, sel.get("ar_steps"), "ar_steps"):
            continue
        out.append(r)

    if str(sel.get("epochs")).lower() == "latest":
        out = _keep_latest_epoch(out)
    return out


def annotate_truth(records: list[FileRecord], cfg: dict) -> list[FileRecord]:
    declared = cfg["io"].get("has_ground_truth", "auto")
    if isinstance(declared, bool):
        return [replace(r, has_truth=declared) for r in records]
    if isinstance(declared, str) and declared.lower() in ("true", "false"):
        val = declared.lower() == "true"
        return [replace(r, has_truth=val) for r in records]
    return [replace(r, has_truth=_detect_truth(r.path)) for r in records]


def cap_files(records: list[FileRecord], cfg: dict) -> list[FileRecord]:
    sel = cfg["select"]
    per_inst = sel.get("max_files_per_instrument")
    total = sel.get("max_files")

    kept = records
    if per_inst:
        counts: dict[str, int] = {}
        kept = []
        for r in records:
            n = counts.get(r.instrument, 0)
            if n < int(per_inst):
                kept.append(r)
                counts[r.instrument] = n + 1
        if len(kept) < len(records):
            print(f"[WARN] max_files_per_instrument={per_inst} trimmed "
                  f"{len(records) - len(kept)} file(s).")

    if total is not None and len(kept) > int(total):
        # Round-robin across instruments so the cap does not land entirely on
        # whichever instrument sorts first.
        by_inst: dict[str, list[FileRecord]] = {}
        for r in kept:
            by_inst.setdefault(r.instrument, []).append(r)
        picked, i = [], 0
        while len(picked) < int(total):
            progressed = False
            for inst in sorted(by_inst):
                if i < len(by_inst[inst]) and len(picked) < int(total):
                    picked.append(by_inst[inst][i])
                    progressed = True
            if not progressed:
                break
            i += 1
        print(
            f"[WARN] {len(kept)} files matched the selection but max_files="
            f"{total}; plotting the first {len(picked)}.\n"
            f"       Narrow select.dates / select.ar_steps / instruments.enabled, "
            f"or set max_files: null to plot everything.\n"
            f"       Run with --filedetection to inspect the full match list."
        )
        kept = picked

    return kept


def _fmt_ranges(vals: list) -> str:
    """Collapse [11,12,13,14,20] -> '11-14, 20'."""
    nums = sorted({int(v) for v in vals if v is not None})
    if not nums:
        return "-"
    runs, start, prev = [], nums[0], nums[0]
    for v in nums[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def print_manifest(records: list[FileRecord], capped: list[FileRecord],
                   verbose: bool = False, row_limit: int = 40) -> None:
    print("=" * 86)
    print(f"File detection: {len(records)} file(s) matched the selection")
    print("=" * 86)
    if not records:
        print("  (nothing found -- check io.data_dir and select.dates)")
        return

    kept_paths = {r.path for r in capped}

    if verbose or len(records) <= row_limit:
        print(f"{'':<2}{'instrument':<20}{'init':<12}{'ar':>3}{'fhr':>5}"
              f"{'ep':>6}{'ba':>4}  {'truth':<6} file")
        print("-" * 86)
        for r in records:
            flag = " " if r.path in kept_paths else "x"
            truth = {True: "yes", False: "no", None: "?"}[r.has_truth]
            name = r.instrument + ("/" + r.variant if r.variant else "")
            print(
                f"{flag:<2}{name:<20}{r.init_time or '-':<12}"
                f"{('-' if r.ar_step is None else r.ar_step):>3}"
                f"{('-' if r.fhr is None else r.fhr):>5}"
                f"{('-' if r.epoch is None else r.epoch):>6}"
                f"{('-' if r.batch is None else r.batch):>4}  {truth:<6} {r.basename}"
            )
    else:
        # Too many to list. Group so the shape of the selection is still clear.
        print(f"(grouped -- {len(records)} rows; pass --verbose for the full list)")
        print(f"{'instrument':<20}{'init':<12}{'ar':>3}{'fhr':>5}"
              f"{'n':>7}  {'truth':<6} epochs")
        print("-" * 86)
        groups: dict[tuple, list[FileRecord]] = {}
        for r in records:
            key = (r.instrument + ("/" + r.variant if r.variant else ""),
                   r.init_time, r.ar_step, r.fhr)
            groups.setdefault(key, []).append(r)
        for (name, init, ar, fhr), rs in sorted(
                groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "",
                                                kv[0][2] or -1, kv[0][3] or -1)):
            truths = {r.has_truth for r in rs}
            truth = ("yes" if truths == {True} else
                     "no" if truths == {False} else "mixed")
            eps = _fmt_ranges([r.epoch for r in rs])
            n_kept = sum(1 for r in rs if r.path in kept_paths)
            marker = f"{len(rs)}" if n_kept == len(rs) else f"{n_kept}/{len(rs)}"
            print(
                f"{name:<20}{init or '-':<12}"
                f"{('-' if ar is None else ar):>3}"
                f"{('-' if fhr is None else fhr):>5}"
                f"{marker:>7}  {truth:<6} {eps[:36]}"
                f"{' ...' if len(eps) > 36 else ''}"
            )

    print("-" * 86)
    by_inst: dict[str, int] = {}
    for r in records:
        by_inst[r.instrument] = by_inst.get(r.instrument, 0) + 1
    print("  by instrument: " + ", ".join(f"{k}={v}" for k, v in sorted(by_inst.items())))

    ar_steps = sorted({r.ar_step for r in records if r.ar_step is not None})
    if ar_steps:
        print(f"  AR steps present: {ar_steps}")
    fhrs = sorted({r.fhr for r in records if r.fhr is not None})
    if fhrs:
        print(f"  forecast hours:   {_fmt_ranges(fhrs)}")
    eps = sorted({r.epoch for r in records if r.epoch is not None})
    if eps:
        print(f"  epochs:           {_fmt_ranges(eps)}")
        if len(eps) > 5:
            print("                    (set select.epochs to a value or range, "
                  "or 'latest', to narrow)")

    print(f"  would plot {len(capped)} of {len(records)} "
          f"('x' marks files excluded by max_files)")
    print("=" * 86)


# =============================================================================
# Driver
# =============================================================================

def figure_plan(rec: FileRecord, spec: InstrumentSpec, cfg: dict) -> list[str]:
    """Which figure families to run for this file."""
    figs = cfg["figures"]
    plan: list[str] = []

    pred_only = figs.get("pred_only", "auto")
    want_pred_only = (
        (pred_only is True)
        or (str(pred_only).lower() == "auto" and rec.has_truth is False)
    )
    if want_pred_only:
        plan.append("pred_only")

    if rec.has_truth:
        for name in spec.figures:
            if figs.get(name, True):
                plan.append(name)
    return plan


def output_dir_for(rec: FileRecord, cfg: dict) -> str:
    """Where this file's figures go: the source subdirectory, mirrored.

    Not optional. obs-space, obs-space-ar0 ... obs-space-ar5, mesh-grid and
    mesh-grid-ar0 ... all hold the same instrument/init/epoch/batch
    combinations, so a flat output directory would have them overwrite each
    other with no warning. Files sitting directly in data_dir fall back to
    ar<N> when an AR step was detected, and to plot_dir otherwise.
    """
    base = os.path.abspath(cfg["io"]["plot_dir"])
    if rec.rel_dir:
        return os.path.join(base, rec.rel_dir)
    if rec.ar_step is not None:
        return os.path.join(base, f"ar{rec.ar_step}")
    return base


def base_tags_for(rec: FileRecord) -> tuple[str, str]:
    fn, tt = "", ""
    if rec.variant:
        fn += f"_{rec.variant}"
        tt += f" - {rec.variant}"
    if rec.init_time:
        fn += f"_init_{rec.init_time}"
        tt += f" - Init {rec.init_time}"
    if rec.epoch is not None:
        fn += f"_epoch_{rec.epoch}"
        tt += f" - Epoch {rec.epoch}"
    if rec.fhr is not None:
        # Mesh dumps carry the forecast hour in the filename rather than in a
        # lead_hours_nominal column; without this every fhr overwrites the last.
        fn += f"_f{rec.fhr:03d}"
        tt += f" - F{rec.fhr:03d}"
    if rec.ar_step is not None:
        fn += f"_ar{rec.ar_step}"
        tt += f" - AR{rec.ar_step}"
    return fn, tt


def prescan_limits(records: list[FileRecord], cfg: dict) -> dict:
    """Compute one set of colour limits per (instrument, feature) up front.

    Without this, AR0 and AR3 maps each rescale to their own data and cannot be
    compared by eye. Reads only the columns it needs.
    """
    import eval_plots as ep

    if not cfg["limits"].get("share_across_ar_steps", True):
        return {}
    if len(records) < 2:
        return {}

    acc: dict[tuple, list[np.ndarray]] = {}
    for rec in records:
        if not rec.has_truth:
            continue
        try:
            df = pd.read_csv(rec.path)
        except Exception:
            continue
        spec = REGISTRY[rec.instrument]
        for feat in ep.discover_features(df, spec.n_channels):
            valid = ep.apply_qc(df, rec.instrument, feat, need_truth=True, cfg=cfg)
            if not np.any(valid):
                continue
            t = ep.np_(df[f"true_{feat}"])[valid]
            p = ep.np_(df[f"pred_{feat}"])[valid]
            acc.setdefault((rec.instrument, feat, "value"), []).append(
                np.concatenate([t, p]))
            acc.setdefault((rec.instrument, feat, "error"), []).append(p - t)

    q = float(cfg["limits"].get("robust_percentile", 99))
    mode = str(cfg["limits"].get("mode", "robust")).lower()
    out: dict[tuple, tuple[float, float]] = {}
    for key, chunks in acc.items():
        allv = np.concatenate(chunks)
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            continue
        if key[2] == "error":
            out[key] = ep.robust_sym_limits(allv, q=q)
        elif mode == "robust":
            lo = float(np.nanpercentile(allv, 100.0 - q))
            hi = float(np.nanpercentile(allv, q))
            out[key] = (lo, hi if hi != lo else lo + 1.0)
        else:
            out[key] = (float(np.nanmin(allv)), float(np.nanmax(allv)))

    if out:
        print(f"[limits] Precomputed shared limits for {len(out) // 2} "
              f"instrument/feature pairs across {len(records)} file(s).")
    return out


def run(cfg: dict, records: list[FileRecord]) -> None:
    import eval_plots as ep

    shared = prescan_limits(records, cfg)
    metrics_rows: list[dict] = []
    n_ok = n_skip = 0
    skipped: list[tuple[str, str]] = []

    for rec in records:
        spec = REGISTRY[rec.instrument]
        chans = (cfg["instruments"].get("channels") or {}).get(rec.instrument)
        spec = replace(spec, channels=tuple(chans) if chans else ())

        plan = figure_plan(rec, spec, cfg)
        if not plan:
            n_skip += 1
            skipped.append((rec.basename, "no applicable figure family"))
            continue

        print(f"\n--- {rec.instrument} | {rec.basename} "
              f"| ar={rec.ar_step} | truth={rec.has_truth} ---")

        try:
            df = pd.read_csv(rec.path)
        except Exception as e:
            n_skip += 1
            skipped.append((rec.basename, f"unreadable ({e})"))
            continue

        if df.empty:
            n_skip += 1
            skipped.append((rec.basename, "empty"))
            continue

        fn_tag, tt_tag = base_tags_for(rec)
        ctx = ep.PlotCtx(
            df=df,
            instrument=rec.instrument,
            spec=spec,
            fig_dir=output_dir_for(rec, cfg),
            base_filename_tag=fn_tag,
            base_title_tag=tt_tag,
            ar_step=rec.ar_step,
            cfg=cfg,
            shared_limits=shared,
            metrics_rows=metrics_rows,
        )

        for fig_name in plan:
            fn = ep.FIGURE_FNS.get(fig_name)
            if fn is None:
                continue
            try:
                fn(ctx)
            except Exception as e:
                print(f"  [ERROR] {fig_name} failed for {rec.basename}: {e}")
        n_ok += 1

    plot_dir = os.path.abspath(cfg["io"]["plot_dir"])
    metrics_df = pd.DataFrame(metrics_rows)

    if cfg["figures"].get("metrics_table", True) and not metrics_df.empty:
        os.makedirs(plot_dir, exist_ok=True)
        out = os.path.join(plot_dir, "metrics.csv")
        metrics_df.to_csv(out, index=False)
        print(f"\n[metrics] Wrote {out} ({len(metrics_df)} rows)")

    if cfg["figures"].get("ar_growth", True) and not metrics_df.empty:
        try:
            ep.plot_ar_growth(metrics_df, os.path.join(plot_dir, "ar_growth"), cfg)
        except Exception as e:
            print(f"[ERROR] ar_growth failed: {e}")

    print("\n" + "=" * 78)
    print(f"Processed {n_ok} file(s); skipped {n_skip}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    print(f"Figures written under: {plot_dir}")
    print("=" * 78)


# =============================================================================
# Metrics-only mode
# =============================================================================

def compute_metrics(records: list[FileRecord], out_path: str,
                    groupby_keys: Sequence[str], min_count: int) -> None:
    """Aggregate RMSE / MAE / bias by summing sufficient statistics.

    Summing n, sum|e|, sum e^2 and sum e before dividing keeps the aggregate
    honest across files and batches, rather than averaging per-file averages.
    Also picks up persist_* columns as a persistence baseline.
    """
    rows_out = []
    for rec in records:
        try:
            df = pd.read_csv(rec.path)
        except Exception as e:
            print(f"[metrics] Skipping unreadable CSV: {rec.path} ({e})")
            continue

        df["instrument"] = rec.instrument
        df["ar_step"] = rec.ar_step if rec.ar_step is not None else -1

        if "lead_hours_nominal" not in df.columns and \
                {"init_time_unix", "valid_time_unix"}.issubset(df.columns):
            try:
                df["lead_hours_nominal"] = (
                    df["valid_time_unix"].astype("int64")
                    - df["init_time_unix"].astype("int64")
                ) / 3600.0
            except Exception:
                pass

        forecast_cols = [("ocelot", c, c[len("pred_"):])
                         for c in df.columns if c.startswith("pred_")]
        forecast_cols += [("persistence", c, c[len("persist_"):])
                          for c in df.columns if c.startswith("persist_")]

        for baseline, pred_col, var in forecast_cols:
            true_col = f"true_{var}"
            if true_col not in df.columns:
                continue
            p = df[pred_col].to_numpy(dtype=float, na_value=np.nan)
            t = df[true_col].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(p) & np.isfinite(t)
            mask_col = f"mask_{var}"
            if mask_col in df.columns:
                valid &= df[mask_col].fillna(False).astype(bool).to_numpy()
            if not valid.any():
                continue

            err = (p[valid] - t[valid]).astype(np.float64)
            n = int(valid.sum())

            gcols = {}
            for k in groupby_keys:
                if k in df.columns:
                    vals = df.loc[valid, k].to_numpy()
                    if k == "lead_hours_nominal":
                        try:
                            vals = vals.astype(float)
                        except Exception:
                            pass
                    gcols[k] = vals
                elif k == "variable":
                    gcols[k] = np.array([var] * n, dtype=object)
                else:
                    gcols[k] = np.array(["unknown"] * n, dtype=object)
            gcols["variable"] = np.array([var] * n, dtype=object)
            gcols["baseline"] = np.array([baseline] * n, dtype=object)

            gdf = pd.DataFrame(gcols)
            gdf["abs_err"] = np.abs(err)
            gdf["sq_err"] = err * err
            gdf["err"] = err

            keys = [k for k in groupby_keys if k not in ("variable", "baseline")]
            agg = gdf.groupby(keys + ["variable", "baseline"], dropna=False).agg(
                n=("err", "size"), sum_abs=("abs_err", "sum"),
                sum_sq=("sq_err", "sum"), sum_err=("err", "sum"),
            ).reset_index()
            if len(agg):
                rows_out.append(agg)

    if not rows_out:
        raise RuntimeError(
            "No metrics produced; check that the CSVs contain pred_*/true_* columns.")

    out_df = pd.concat(rows_out, ignore_index=True)
    gb = [k for k in groupby_keys if k not in ("variable", "baseline")
          and k in out_df.columns] + ["variable", "baseline"]
    out_df = out_df.groupby(gb, dropna=False, as_index=False).agg(
        n=("n", "sum"), sum_abs=("sum_abs", "sum"),
        sum_sq=("sum_sq", "sum"), sum_err=("sum_err", "sum"),
    )
    out_df = out_df[out_df["n"] >= int(min_count)].copy()
    denom = out_df["n"].astype(float).replace(0.0, np.nan)
    out_df["mae"] = out_df["sum_abs"] / denom
    out_df["bias"] = out_df["sum_err"] / denom
    out_df["rmse"] = np.sqrt(out_df["sum_sq"] / denom)
    out_df.drop(columns=["sum_abs", "sum_sq", "sum_err"], inplace=True)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[metrics] Wrote: {out_path} (rows={len(out_df)})")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OCELOT evaluation plots and metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to plotting.yaml. Pass '' for built-in defaults.")
    p.add_argument("--mode", choices=["plots", "metrics"], default="plots")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="With --filedetection, list every file instead of grouping.")
    p.add_argument("--filedetection", "--dry-run", action="store_true",
                   dest="filedetection",
                   help="List the files that match and exit without plotting.")

    p.add_argument("--base-dir", "--base_dir", dest="base_dir", default=None,
                   help="Prefix for relative data_dir/plot_dir in the config.")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", default=None)
    p.add_argument("--plot-dir", "--plot_dir", dest="plot_dir", default=None)
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--dates", nargs="*", default=None,
                   help="One init time, or START END for an inclusive range.")
    p.add_argument("--ar-steps", "--ar_steps", dest="ar_steps", nargs="*", default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--batch-idx", "--batch_idx", dest="batch_idx", type=int, default=None)
    p.add_argument("--fhr", type=int, default=None)
    p.add_argument("--subwindow-leads", "--subwindow_leads", dest="subwindow_leads",
                   nargs="*", default=None,
                   help="Lead hours naming the END of each window, e.g. 3 9 "
                        "for 0-3h and 6-9h.")
    p.add_argument("--no-horizons", dest="no_horizons", action="store_true")
    p.add_argument("--max-files", "--max_files", dest="max_files", type=int, default=None,
                   help="Cap on files to plot. Use -1 for no cap.")
    p.add_argument("--has-ground-truth", "--has_ground_truth",
                   dest="has_ground_truth", default=None,
                   choices=["auto", "true", "false"])
    p.add_argument("--strict-obs-window", "--strict_obs_window",
                   dest="strict_obs_window", action="store_true")

    p.add_argument("--print-config", dest="print_config", action="store_true",
                   help="Dump the effective config after CLI overrides, then exit.")
    p.add_argument("--metrics-out", dest="metrics_out", default="metrics_pointwise.csv")
    p.add_argument("--metrics-groupby", dest="metrics_groupby",
                   default="instrument,ar_step,lead_hours_nominal")
    p.add_argument("--metrics-min-count", dest="metrics_min_count", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Line-buffer stdout. Under some launchers (sbatch, `sh script.sh`) a
    # block-buffered pipe can swallow everything if the process is reaped
    # before the buffer flushes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args(argv)
    print(f"[evaluations] python {sys.version.split()[0]} | {__file__}")
    print(f"[evaluations] config={args.config or '<builtin defaults>'} "
          f"mode={args.mode} filedetection={args.filedetection}")
    sys.stdout.flush()

    cfg = load_config(args.config or None)
    cfg = apply_cli_overrides(cfg, args)
    cfg = resolve_paths(cfg)

    if args.print_config:
        import yaml
        print("# effective configuration after CLI overrides")
        print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        return 0

    dd = cfg["io"]["data_dir"]
    for root in ([dd] if isinstance(dd, str) else list(dd)):
        print(f"[evaluations] scanning {root} "
              f"({'exists' if os.path.isdir(root) else 'MISSING'})")
    sys.stdout.flush()

    unmatched: list = []
    records = scan(cfg, unmatched)
    print(f"[evaluations] discovered {len(records)} candidate file(s), "
          f"{len(unmatched)} unrecognised")
    sys.stdout.flush()
    all_found = list(records)
    records = apply_filters(records, cfg)
    records = annotate_truth(records, cfg)

    if not records:
        print("No CSV files matched the current selection.")
        print(f"  base_dir: {cfg['io'].get('base_dir')}")
        print(f"  data_dir: {cfg['io']['data_dir']}")
        print(f"  dates:    {cfg['select'].get('dates')}")
        print(f"  ar_steps: {cfg['select'].get('ar_steps')}")
        if all_found:
            print(f"\n{len(all_found)} CSV(s) were recognised but filtered out. "
                  f"Present in the tree:")
            insts: dict = {}
            for r in all_found:
                insts.setdefault(r.instrument, set()).add(r.init_time)
            for k in sorted(insts):
                inits = sorted(x for x in insts[k] if x)
                print(f"    {k:<14} init times: {inits[:6]}"
                      f"{' ...' if len(inits) > 6 else ''}")
        if unmatched:
            print(f"\n{len(unmatched)} file(s) skipped during discovery:")
            for path, why in unmatched[:10]:
                print(f"    {os.path.basename(path)}: {why}")
            if len(unmatched) > 10:
                print(f"    ... and {len(unmatched) - 10} more")
        print("\nWiden select.dates / select.epochs / instruments.enabled "
              "and rerun --filedetection.")
        return 1

    print(f"[evaluations] {len(records)} file(s) passed the select filters")
    capped = cap_files(records, cfg)

    if args.filedetection:
        print_manifest(records, capped, verbose=args.verbose)
        sys.stdout.flush()
        return 0

    if args.mode == "metrics":
        keys = [k.strip() for k in args.metrics_groupby.split(",") if k.strip()]
        compute_metrics(capped, args.metrics_out, keys, args.metrics_min_count)
        return 0

    print(f"Selected {len(capped)} of {len(records)} matched file(s).")
    run(cfg, capped)
    return 0


if __name__ == "__main__":
    try:
        _rc = main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.exit(_rc)
