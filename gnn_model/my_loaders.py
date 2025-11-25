import os
import pandas as pd

# -------------------- basic helpers --------------------


def _coerce_cols(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def _date_list(start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    days = int((end - start).days) + 1
    return [start + pd.Timedelta(days=i) for i in range(days)]


def _load_local_partitioned(root, dataset_id, start_date, end_date, columns):
    """
    Local reader for NNJA OBS_DATE=YYYY-MM-DD partitions.
    Scans [start_date, end_date], collects *.parquet, returns DataFrame with requested columns.
    """
    import glob

    base = os.path.join(root, "data", "v1", *dataset_id.split("-"))

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    days = int((end - start).days) + 1

    files = []
    for i in range(days):
        d = start + pd.Timedelta(days=i)
        day_dir = os.path.join(base, f"OBS_DATE={d.date().isoformat()}")
        if os.path.isdir(day_dir):
            files.extend(sorted(glob.glob(os.path.join(day_dir, "*.parquet"))))

    if not files:
        return pd.DataFrame(columns=columns)

    parts = []
    for fp in files:
        try:
            parts.append(pd.read_parquet(fp, columns=columns))
        except Exception:
            # if column-pruning fails (e.g., some columns missing), read all then subset
            df = pd.read_parquet(fp)
            keep = [c for c in columns if c in df.columns]
            parts.append(df[keep])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    # ensure all requested columns exist
    for c in columns:
        if c not in out.columns:
            out[c] = pd.Series(dtype="float64")

    return out


def _maybe_local(dataset_id, start_date, end_date, columns):
    root = os.getenv("NNJA_LOCAL_ROOT")
    if root:
        return _load_local_partitioned(root, dataset_id, start_date, end_date, columns)
    return None


# -------------------- schema unification --------------------

# Canonical column names we expose downstream (NC000101 style).
# These align with your var_map + coords you pass to the adapter.
CANONICAL = [
    # coords/time/meta
    "OBS_TIMESTAMP",
    "LAT",
    "LON",
    "SELV",
    "RPID",
    # core variables (101-style)
    "PRESDATA.PRESSQ03.PRES",  # pressure (Pa)
    "QMPR",  # pressure QM
    "TEMHUMDA.TMDB",  # air temperature (K)
    "QMAT",  # temp QM
    "TEMHUMDA.TMDP",  # dewpoint (K)
    "QMDD",  # dewpoint QM
    "TEMHUMDA.REHU",  # RH (%)
    "BSYWND1.WSPD",  # wind speed (m/s)
    "BSYWND1.WDIR",  # wind dir (deg)
    "QMWN",  # wind QM
]

# Map NC000001 columns -> canonical NC000101 names
ALIASES_NC000001_TO_101 = {
    # pressure
    "PRSSQ1.PRES": "PRESDATA.PRESSQ03.PRES",
    "PRSSQ1.QMPR": "QMPR",
    # temperature & dewpoint
    "TMPSQ1.TMDB": "TEMHUMDA.TMDB",
    "TMPSQ1.TMDP": "TEMHUMDA.TMDP",
    "TMPSQ1.QMAT": "QMAT",
    "TMPSQ1.QMDD": "QMDD",
    # humidity
    "TMPSQ1.TMPSQ2.REHU": "TEMHUMDA.REHU",
    # wind
    "WNDSQ1.WSPD": "BSYWND1.WSPD",
    "WNDSQ1.WDIR": "BSYWND1.WDIR",
    "WNDSQ1.QMWN": "QMWN",
    # coords/time/meta (same names kept for clarity)
    "OBS_TIMESTAMP": "OBS_TIMESTAMP",
    "LAT": "LAT",
    "LON": "LON",
    "SELV": "SELV",
    "RPID": "RPID",
}

# Coords we always try to include (RPID helps de-dup)
REQUIRED_COORDS = ["OBS_TIMESTAMP", "LAT", "LON", "SELV", "RPID"]


def _columns_for_dataset(dataset_id, want_canonical):
    """
    Build the minimal set of columns to request from a given dataset,
    including alias sources if we're reading NC000001.
    """
    want_canon = list(dict.fromkeys(list(want_canonical)))

    if str(dataset_id).endswith("NC000001"):
        # For NC000001, request coords in canonical form (they exist with same names)
        base = [c for c in want_canon if c in ("OBS_TIMESTAMP", "LAT", "LON", "SELV", "RPID")]
        # and request ONLY alias sources for data/QM columns
        alias_srcs = [src for src, dst in ALIASES_NC000001_TO_101.items() if dst in want_canon]
        return sorted(set(base + alias_srcs))

    # NC000101: just request the canonical columns
    return want_canon


def _rename_aliases_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    to_rename = {src: dst for src, dst in ALIASES_NC000001_TO_101.items() if src in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
    # ensure unique column labels (protects against unexpected collisions)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


# -------------------- reading (local → cloud) --------------------


def _read_one_dataset(dataset_id, start_date, end_date, want_for_ds, mirror):
    """
    Try local → cloud for one dataset id, return DataFrame with at least the columns we asked for.
    """
    # local (preferred)
    df = _maybe_local(dataset_id, start_date, end_date, want_for_ds)
    if df is not None:
        df["__src"] = dataset_id
        return df

    # cloud (DataCatalog) fallback
    from nnja_ai import DataCatalog

    catalog = DataCatalog(mirror=mirror)
    ds = catalog[dataset_id].sel(time=slice(str(start_date), str(end_date)))
    # Try requesting only the variables we want; if that fails, load all and subset.
    try:
        ds2 = ds.sel(variables=[c for c in want_for_ds if c in ds.variables])
        df = ds2.load_dataset(backend="pandas")
    except Exception:
        df = ds.load_dataset(backend="pandas")
        keep = [c for c in want_for_ds if c in df.columns]
        df = df[keep] if keep else df

    df["__src"] = dataset_id
    return df


# -------------------- QM normalization & de-dup preference --------------------

QM_CANON = ["QMPR", "QMAT", "QMDD", "QMWN"]


def _normalize_qm_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make QM semantics consistent across mirrors:
      - cast numeric, NaN -> -1 (unknown)
      - NC000001 uses -9999 for missing: map to -1
    """
    for qm in QM_CANON:
        if qm in df.columns:
            s = pd.to_numeric(df[qm], errors="coerce").fillna(-1)
            s = s.mask(s == -9999, -1)
            df[qm] = s.astype("int16")
    return df


def _nnz_targets(df: pd.DataFrame) -> pd.Series:
    # targets we care about for preference (available columns only)
    targets = [
        "PRESDATA.PRESSQ03.PRES",
        "TEMHUMDA.TMDB",
        "TEMHUMDA.TMDP",
        "TEMHUMDA.REHU",
        "BSYWND1.WSPD",
        "BSYWND1.WDIR",
    ]
    have = [c for c in targets if c in df.columns]
    if not have:
        return pd.Series(0, index=df.index)
    return df[have].notna().sum(axis=1)


def _dedup_with_preference(df: pd.DataFrame) -> pd.DataFrame:
    """
    When both schemas contribute rows for the same observation time/station (or near-coincident
    lat/lon), pick **one** row using a scoring policy:
      1) Avoid QM=14 if any QM column has it (heavy penalty)
      2) Prefer NC000101 over NC000001
      3) Prefer rows with more populated target fields
    """
    # ensure source marker exists
    if "__src" not in df.columns:
        df["__src"] = "unknown"

    qm_cols = [c for c in QM_CANON if c in df.columns]
    has_bad14 = df[qm_cols].eq(14).any(axis=1) if qm_cols else False

    src_weight = {
        "conv-adpsfc-NC000101": 5,  # ML-flattened
        "conv-adpsfc-NC000001": 4,  # BUFR (land synop)
        "conv-adpsfc-NC000007": 3,  # METAR/SPECI
        "conv-adpsfc-NC000002": 2,  # SYNOP mobile
    }

    df["__score"] = (
        -100 * has_bad14.astype(int) + df["__src"].map(src_weight).fillna(0).astype(int) + _nnz_targets(df)  # prefer rows with more populated targets
    )

    # grouping keys: prefer (time, RPID); else ~coincident lat/lon
    if "RPID" in df.columns:
        keys = ["OBS_TIMESTAMP", "RPID"]
    else:
        df = df.assign(__latr=df["LAT"].round(1), __lonr=df["LON"].round(1))
        keys = ["OBS_TIMESTAMP", "__latr", "__lonr"]

    keep_idx = df.groupby(keys, dropna=False)["__score"].idxmax()
    out = df.loc[keep_idx].drop(columns=[c for c in ["__score", "__latr", "__lonr"] if c in df.columns])
    return out.reset_index(drop=True)


# -------------------- public loaders --------------------


def load_adpsfc(
    start_date,
    end_date,
    columns,
    dataset_ids=("conv-adpsfc-NC000101", "conv-adpsfc-NC000001", "conv-adpsfc-NC000007", "conv-adpsfc-NC000002"),  # + METAR, + SYNOP mobile
    mirror="gcp_brightband",
    drop_dupes=True,
):
    """
    Load surface conventional obs from one or more NNJA datasets, unify schema to NC000101,
    normalize QM flags, and (optionally) de-duplicate overlaps with a preference policy.

    Parameters
    ----------
    start_date, end_date : str or datetime-like
    columns              : iterable of canonical column names you want (NC000101 style)
    dataset_ids          : tuple/list or str of dataset IDs to read
    mirror               : DataCatalog mirror name
    drop_dupes           : if True, apply preference-based de-duplication
    """
    # canonical columns we will expose downstream (ensure coords)
    want_canon = list(dict.fromkeys(list(columns) + REQUIRED_COORDS))

    # normalize dataset_ids
    if isinstance(dataset_ids, str):
        dataset_ids = (dataset_ids,)

    # read & unify each dataset
    dfs = []
    for dsid in dataset_ids:
        want_for_ds = _columns_for_dataset(dsid, want_canon)
        df = _read_one_dataset(dsid, start_date, end_date, want_for_ds, mirror)
        if df is None or df.empty:
            continue

        # rename NC000001 aliases -> canonical
        df = _rename_aliases_to_canonical(df)

        # normalize QM flags
        df = _normalize_qm_flags(df)

        # backfill any missing canonical columns
        for c in want_canon:
            if c not in df.columns:
                df[c] = pd.Series(dtype="float64")

        # final subset & order
        df = _coerce_cols(df, want_canon + ["__src"])  # keep __src for de-dup
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=want_canon)

    out = pd.concat(dfs, ignore_index=True)

    # optional de-duplication with preference
    if drop_dupes:
        out = _dedup_with_preference(out)

    # drop helper column if present
    if "__src" in out.columns:
        out = out.drop(columns="__src")

    return _coerce_cols(out, want_canon)


def load_adpupa(start_date, end_date, columns, message="conv-adpupa-NC002001", mirror="gcp_brightband"):
    """
    Radiosonde path unchanged; local if NNJA_LOCAL_ROOT is set, else DataCatalog.
    """
    df = _maybe_local(message, start_date, end_date, columns)
    if df is not None:
        return _coerce_cols(df, columns)

    from nnja_ai import DataCatalog

    catalog = DataCatalog(mirror=mirror)
    ds = catalog[message].sel(time=slice(str(start_date), str(end_date)), variables=columns)
    df = ds.load_dataset(backend="pandas")
    return _coerce_cols(df, columns)
