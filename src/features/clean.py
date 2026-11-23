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
    """Merge a fetch into the stored table: newest fetch wins, but an empty value never wins.

    A value the provider corrected replaces the old one. A value the API returns
    empty (NaN, or a column `validate` dropped because all re-fetched days were
    empty) keeps the stored one (editorial B3): labels and vwc_latest come from
    this table, so a short API outage must not delete them. Each kept value is
    logged, because a value the provider deleted on purpose would also stay.
    Running the same fetch twice gives the same table (idempotent).
    """
    if old is None or old.empty:
        return new.sort_index()
    overlap = new.index.intersection(old.index)
    stored = old.loc[overlap]
    for col in old.columns:
        fetched = new.loc[overlap, col] if col in new else pd.Series(float("nan"), index=overlap)
        kept = fetched.isna() & stored[col].notna()
        if kept.any():
            log.warning("%s: API returned no value for %d stored date(s), kept the stored value: %s",
                        col, int(kept.sum()), [d.date().isoformat() for d in overlap[kept.to_numpy()]])
    columns = list(new.columns) + [c for c in old.columns if c not in new.columns]
    return new.combine_first(old)[columns].sort_index()


def upsert_forecasts(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge forecasts, but here the FIRST value written wins.

    Unlike observations, a forecast must stay as it was when it was issued
    (the proposal: live evaluation keeps the original inputs). So new values
    only fill gaps; they never overwrite an existing value.
    """
    if old is None or old.empty:
        return new.sort_index()
    return old.combine_first(new).sort_index()
