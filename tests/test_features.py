"""Tests for common/features.py, the feature definitions shared by all pipelines.

Horizon: a forecast is issued on day D using data through D-1; the target is
the daily mean soil moisture of D+1; persistence predicts obs(D-1).
So for target d, features may only use observations dated <= d-2.
"""
import numpy as np
import pandas as pd
import pytest

from common.features import FEATURE_COLUMNS, build_feature_rows


def make_obs(n=30, start="2025-01-01"):
    """Small series with distinct values so a wrong shift is easy to spot."""
    idx = pd.date_range(start, periods=n, freq="D", name="date")
    i = np.arange(n, dtype=float)
    return pd.DataFrame(
        {"cosmos_vwc": 20 + i, "precip": i % 5, "ta": 5 + i / 10, "rh": 80 - i / 10, "stp_tsoil20": 6 + i / 20},
        index=idx,
    )


def test_no_future_data_hand_computed():
    """Every feature for target d equals a value computed by hand from obs dated <= d-2."""
    obs = make_obs()
    f = build_feature_rows(obs)
    d = pd.Timestamp("2025-01-20")
    L = d - pd.Timedelta(days=2)  # latest observation known at issue time

    def at(col, day):
        return obs.loc[day, col]

    def days(k):  # the k days ending at L
        return pd.date_range(L - pd.Timedelta(days=k - 1), L)

    row = f.loc[d]
    assert row["vwc_latest"] == at("cosmos_vwc", L)
    assert row["vwc_lag1"] == at("cosmos_vwc", L - pd.Timedelta(days=1))
    assert row["vwc_lag3"] == at("cosmos_vwc", L - pd.Timedelta(days=3))
    assert row["vwc_lag7"] == at("cosmos_vwc", L - pd.Timedelta(days=7))
    assert row["vwc_mean3"] == pytest.approx(obs.loc[days(3), "cosmos_vwc"].mean())
    assert row["vwc_mean7"] == pytest.approx(obs.loc[days(7), "cosmos_vwc"].mean())
    assert row["precip_sum3"] == pytest.approx(obs.loc[days(3), "precip"].sum())
    assert row["precip_sum7"] == pytest.approx(obs.loc[days(7), "precip"].sum())
    for c in ["ta", "rh", "stp_tsoil20"]:
        assert row[c] == at(c, L)
    assert row["day_of_year"] == d.day_of_year
    assert row["issue_date"] == d - pd.Timedelta(days=1)
    assert row["cosmos_vwc"] == at("cosmos_vwc", d)  # the label IS obs(d)


def test_changing_recent_observations_does_not_change_features():
    """obs(d-1) and obs(d) are unknown at issue time: changing them must not move any feature."""
    obs = make_obs()
    d = pd.Timestamp("2025-01-20")
    before = build_feature_rows(obs).loc[d, FEATURE_COLUMNS]

    changed = obs.copy()
    changed.loc[[d - pd.Timedelta(days=1), d]] = 999.0
    after = build_feature_rows(changed).loc[d, FEATURE_COLUMNS]

    pd.testing.assert_series_equal(before, after)


def test_gap_gives_nan_not_shifted_value():
    """With a day missing from the input, shifts are by calendar day, so lags become NaN."""
    obs = make_obs().drop(pd.Timestamp("2025-01-10"))  # day completely absent
    f = build_feature_rows(obs)

    # Target 2025-01-12 has L = 2025-01-10, the missing day.
    assert np.isnan(f.loc["2025-01-12", "vwc_latest"])
    assert not f.loc["2025-01-12", "inference_ok"]
    # Target 2025-01-13 has lag1 = 2025-01-10 -> NaN, but vwc_latest (01-11) exists.
    assert np.isnan(f.loc["2025-01-13", "vwc_lag1"])
    assert f.loc["2025-01-13", "vwc_latest"] == obs.loc["2025-01-11", "cosmos_vwc"]
    # A window containing the gap is NaN (no partial means / sums).
    assert np.isnan(f.loc["2025-01-14", "vwc_mean3"])
    assert np.isnan(f.loc["2025-01-14", "precip_sum7"])


def test_missing_value_is_not_forward_filled():
    obs = make_obs()
    obs.loc["2025-01-10", "cosmos_vwc"] = np.nan  # present row, NaN value
    f = build_feature_rows(obs)
    assert np.isnan(f.loc["2025-01-12", "vwc_latest"])
    assert not f.loc["2025-01-12", "inference_ok"]
    assert not f.loc["2025-01-10", "is_labelled"]


def test_unobserved_target_rows_have_nan_label_and_fill_later():
    obs = make_obs(n=20)  # last observation 2025-01-20
    last_target = pd.Timestamp("2025-01-22")  # "today + 1" when today = 01-21
    f = build_feature_rows(obs, last_target=last_target)

    assert f.index.max() == last_target
    assert np.isnan(f.loc[last_target, "cosmos_vwc"])
    assert not f.loc[last_target, "is_labelled"]
    assert f.loc[last_target, "inference_ok"]  # obs(01-20) exists, so we can forecast

    # The next day the observation for 01-22 arrives and the label is filled.
    more = make_obs(n=22)
    f2 = build_feature_rows(more, last_target=last_target)
    assert f2.loc[last_target, "cosmos_vwc"] == more.loc[last_target, "cosmos_vwc"]
    assert f2.loc[last_target, "is_labelled"]


def test_forecast_features_use_issue_day_and_target_day():
    obs = make_obs()
    fc = pd.DataFrame(
        {"precip_fc_day1": np.arange(30, dtype=float), "precip_fc_day2": 100 + np.arange(30, dtype=float),
         "fc_source": "archive"},
        index=obs.index,
    )
    f = build_feature_rows(obs, fc)
    d = pd.Timestamp("2025-01-20")
    assert f.loc[d, "precip_fc_issue_day"] == fc.loc[d - pd.Timedelta(days=1), "precip_fc_day1"]
    assert f.loc[d, "precip_fc_target_day"] == fc.loc[d, "precip_fc_day2"]
    assert f.loc[d, "precip_fc_gap_sum"] == f.loc[d, "precip_fc_issue_day"] + f.loc[d, "precip_fc_target_day"]
    assert f.loc[d, "fc_source"] == "archive"


def test_without_forecasts_columns_exist_as_nan():
    f = build_feature_rows(make_obs())
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert f["precip_fc_gap_sum"].isna().all()
