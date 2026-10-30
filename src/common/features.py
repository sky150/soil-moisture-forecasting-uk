"""Feature definitions: the ONE place where model inputs are computed.

The feature pipeline, the training pipeline (MS3) and the inference pipeline
(MS4) all call `build_feature_rows`, so a feature is always computed the same
way in training and in serving (no training-serving skew).

Forecast horizon (editorial point E3):
    A forecast is issued on day D using data through D-1; the target is the
    daily mean soil moisture of D+1; persistence predicts obs(D-1).

In this module one row = one target date d = D+1. The latest observation that
exists at issue time is L = d - 2 (= D-1). Every observation-based feature
therefore only uses data dated <= d - 2, and persistence for target d is obs(d-2)
(= the column `vwc_latest`).

"Lag k" means k days before L (so vwc_lag1 = obs(d-3)). This is a choice made
when designing the pipeline, not something from the proposal; it keeps
vwc_latest and the lags from being the same value.
"""
from __future__ import annotations

import pandas as pd

OBS_COLUMNS = ["cosmos_vwc", "precip", "ta", "rh", "stp_tsoil20"]
LABEL = "cosmos_vwc"

# Model inputs, in a fixed order. Training and inference select exactly these.
FEATURE_COLUMNS = [
    "vwc_latest",
    "vwc_lag1",
    "vwc_lag3",
    "vwc_lag7",
    "vwc_mean3",
    "vwc_mean7",
    "precip_sum3",
    "precip_sum7",
    "ta",
    "rh",
    "stp_tsoil20",
    "day_of_year",
    "precip_fc_issue_day",
    "precip_fc_target_day",
    "precip_fc_gap_sum",
]


def build_feature_rows(
    observations: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
    issue_gap: int = 2,
    last_target: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Turn daily observations (+ archived/live forecasts) into one row per target date.

    observations: index = calendar date, columns = OBS_COLUMNS (missing ones become NaN).
    forecasts:    index = calendar date, columns precip_fc_day1 (forecast made 1 day
                  before that date), precip_fc_day2 (made 2 days before) and fc_source.
    last_target:  last target date to create a row for (the pipeline passes today + 1,
                  so the row for tomorrow exists with an empty label).
    """
    obs = observations.copy()
    for c in OBS_COLUMNS:
        if c not in obs:
            obs[c] = float("nan")

    # Reindex onto a COMPLETE daily calendar. After this, shift(k) means
    # "k calendar days earlier". Without it, a missing day would make shift()
    # silently take an older value (positional instead of calendar shift).
    start = obs.index.min()
    end = max(obs.index.max(), last_target) if last_target is not None else obs.index.max()
    calendar = pd.date_range(start, end, freq="D", name="target_date")
    obs = obs.reindex(calendar)

    g = issue_gap  # rows are indexed by d; shift(g) gives the value at L = d - g
    vwc = obs["cosmos_vwc"]
    f = pd.DataFrame(index=calendar)
    f["issue_date"] = calendar - pd.Timedelta(days=g - 1)  # day D = d - 1

    f["vwc_latest"] = vwc.shift(g)  # obs(L) = persistence forecast
    f["vwc_lag1"] = vwc.shift(g + 1)  # obs(L-1)
    f["vwc_lag3"] = vwc.shift(g + 3)  # obs(L-3)
    f["vwc_lag7"] = vwc.shift(g + 7)  # obs(L-7)
    # min_periods = full window: if one day in the window is missing the
    # feature is NaN (missing-day rule: never fill in values we did not observe).
    f["vwc_mean3"] = vwc.rolling(3, min_periods=3).mean().shift(g)
    f["vwc_mean7"] = vwc.rolling(7, min_periods=7).mean().shift(g)
    # precip is a daily SUM, so multi-day rain is a sum too (not a mean).
    f["precip_sum3"] = obs["precip"].rolling(3, min_periods=3).sum().shift(g)
    f["precip_sum7"] = obs["precip"].rolling(7, min_periods=7).sum().shift(g)
    for c in ["ta", "rh", "stp_tsoil20"]:
        f[c] = obs[c].shift(g)
    f["day_of_year"] = calendar.day_of_year

    # Forecast rain for the two days nobody has observed at issue time:
    # the issue day D (= d-1) and the target day d. Both values come from a
    # forecast made on D-1, so they already exist when we issue on day D.
    fc = forecasts if forecasts is not None else pd.DataFrame()
    fc = fc.reindex(calendar)
    day1 = fc.get("precip_fc_day1", pd.Series(float("nan"), index=calendar))
    day2 = fc.get("precip_fc_day2", pd.Series(float("nan"), index=calendar))
    f["precip_fc_issue_day"] = day1.shift(1)  # forecast for d-1, made one day before it
    f["precip_fc_target_day"] = day2  # forecast for d, made two days before it
    f["precip_fc_gap_sum"] = f["precip_fc_issue_day"] + f["precip_fc_target_day"]
    f["fc_source"] = fc.get("fc_source", pd.Series(None, index=calendar, dtype="object"))

    # Label: what actually happened on d. NaN until COSMOS publishes it.
    f[LABEL] = vwc
    f["is_labelled"] = f[LABEL].notna()
    # No forecast is issued when the latest observation is missing (missing-day rule).
    f["inference_ok"] = f["vwc_latest"].notna()
    return f
