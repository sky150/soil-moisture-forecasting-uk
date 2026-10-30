"""Validate fetched data and merge it into the clean tables."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def validate(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Remove duplicate dates, set implausible values to NaN, drop empty columns.

    Implausible values become NaN (not dropped rows) so one bad sensor does
    not delete the other measurements of that day. Counts are logged.
    """
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.copy()
    for col, (lo, hi) in ranges.items():
        if col in df:
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                log.warning("%s: %d value(s) outside [%s, %s] set to NaN", col, int(bad.sum()), lo, hi)
            df.loc[bad, col] = float("nan")
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        log.info("Dropping empty columns: %s", empty)
    return df.drop(columns=empty)


def upsert(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Replace the rows for every date in `new` (newest fetch wins), keep the rest.

    Running the same fetch twice gives the same table (idempotent), and a
    value the provider corrected replaces the old one.
    """
    if old is None or old.empty:
        return new.sort_index()
    kept = old[~old.index.isin(new.index)]
    return pd.concat([kept, new]).sort_index()


def upsert_forecasts(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge forecasts, but here the FIRST value written wins.

    Unlike observations, a forecast must stay as it was when it was issued
    (the proposal: live evaluation keeps the original inputs). So new values
    only fill gaps; they never overwrite an existing value.
    """
    if old is None or old.empty:
        return new.sort_index()
    return old.combine_first(new).sort_index()
