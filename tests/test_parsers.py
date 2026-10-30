"""Parsing COSMOS CoverageJSON and Open-Meteo responses (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import cosmos, openmeteo

FIXTURES = Path(__file__).parent / "fixtures"
PARAMS = ["cosmos_vwc", "precip", "ta", "rh", "stp_tsoil20"]


def load(name):
    return json.loads((FIXTURES / name).read_text())


def coverage(times, ranges):
    """Minimal single Coverage in the same shape as the COSMOS API."""
    return {
        "type": "Coverage",
        "domain": {"axes": {"t": {"values": times}}},
        "ranges": {k: {"values": v} for k, v in ranges.items()},
    }


def test_parse_real_collection():
    """Trimmed real response for BALRD, 1-10 March 2025 (COSMOS-UK, Open Government Licence)."""
    df = cosmos.parse(load("cosmos_balrd_2025-03-01_10.json"), PARAMS)
    assert list(df.columns) == PARAMS
    assert len(df) == 10
    assert df.index[0] == pd.Timestamp("2025-03-01")
    assert df.loc["2025-03-01", "cosmos_vwc"] == 38.0
    assert df.loc["2025-03-10", "cosmos_vwc"] == 32.6


def test_parse_single_coverage_with_null_and_absent_parameter():
    js = coverage(
        ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
        {"cosmos_vwc": [30.5, None], "precip": [1.2, 0.0]},  # ta, rh, stp_tsoil20 absent
    )
    df = cosmos.parse(js, PARAMS)
    assert list(df.columns) == PARAMS
    assert df.loc["2025-01-01", "cosmos_vwc"] == 30.5
    assert np.isnan(df.loc["2025-01-02", "cosmos_vwc"])
    assert df["ta"].isna().all()


def test_parse_none_gives_empty_frame():
    df = cosmos.parse(None, PARAMS)
    assert df.empty and list(df.columns) == PARAMS


def test_year_windows_split_on_calendar_years():
    from datetime import date

    w = cosmos.year_windows(date(2023, 6, 1), date(2025, 2, 3))
    assert w == [
        (date(2023, 6, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 2, 3)),
    ]


def test_openmeteo_hourly_to_daily_sum():
    js = load("openmeteo_balrd_2025-03-01_02.json")
    daily = openmeteo.parse_daily(js)
    hours = pd.Series(js["hourly"]["precipitation_previous_day2"][:24], dtype=float)
    assert list(daily.columns) == ["precip_fc_day1", "precip_fc_day2"]
    assert len(daily) == 2
    assert daily.loc["2025-03-01", "precip_fc_day2"] == hours.sum()


def test_openmeteo_incomplete_day_is_nan():
    times = [f"2024-01-01T{h:02d}:00" for h in range(24)]
    js = {"hourly": {"time": times,
                     "precipitation_previous_day1": [0.1] * 23 + [None],
                     "precipitation_previous_day2": [None] * 24}}
    daily = openmeteo.parse_daily(js)
    assert np.isnan(daily.loc["2024-01-01", "precip_fc_day1"])  # 23 of 24 hours is not a day
    assert np.isnan(daily.loc["2024-01-01", "precip_fc_day2"])
