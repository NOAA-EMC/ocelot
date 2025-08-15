import os
import datetime as dt
import pandas as pd


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
    Scans dates in [start_date, end_date], collects *.parquet files,
    and returns a pandas.DataFrame with requested columns.
    """
    import os
    import glob
    import pandas as pd

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
            df = pd.read_parquet(fp)
            keep = [c for c in columns if c in df.columns]
            parts.append(df[keep])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    for c in columns:
        if c not in out.columns:
            out[c] = pd.Series(dtype="float64")
    return out


def _maybe_local(dataset_id, start_date, end_date, columns):
    root = os.getenv("NNJA_LOCAL_ROOT")
    if root:
        return _load_local_partitioned(root, dataset_id, start_date, end_date, columns)
    return None


def load_adpsfc(start_date, end_date, columns, message="conv-adpsfc-NC000101", mirror="gcp_brightband"):
    # Try local first
    df = _maybe_local(message, start_date, end_date, columns)
    if df is not None:
        return _coerce_cols(df, columns)

    # Fallback to cloud (requires outbound internet)
    from nnja_ai import DataCatalog

    catalog = DataCatalog(mirror=mirror)
    ds = catalog[message].sel(time=slice(str(start_date), str(end_date)), variables=columns)
    df = ds.load_dataset(backend="pandas")
    return _coerce_cols(df, columns)


def load_adpupa(start_date, end_date, columns, message="conv-adpupa-NC002001", mirror="gcp_brightband"):
    df = _maybe_local(message, start_date, end_date, columns)
    if df is not None:
        return _coerce_cols(df, columns)

    from nnja_ai import DataCatalog

    catalog = DataCatalog(mirror=mirror)
    ds = catalog[message].sel(time=slice(str(start_date), str(end_date)), variables=columns)
    df = ds.load_dataset(backend="pandas")
    return _coerce_cols(df, columns)
