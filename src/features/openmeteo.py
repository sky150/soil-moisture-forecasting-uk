"""Open-Meteo precipitation forecasts (editorial point E2).

We use the Previous Runs API because it returns what a forecast said a fixed
time in advance: `precipitation_previous_day1` for an hour is the forecast made
one day before that hour, `_previous_day2` two days before. That is exactly
what we know at issue time, both for the historical backfill and live.
(The plain "Historical Forecast API" stitches the first hours of every model
run together, which is close to observed rain and would leak.)

Archive coverage for BALRD, checked on 2026-10-30: empty on 2024-01-01,
complete from 2024-03-01. Earlier dates stay NaN.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from common import http
from common.config import Config

HOURLY = ["precipitation_previous_day1", "precipitation_previous_day2"]


def fetch(cfg: Config, start: date, end: date):
    """Hourly lead-time forecasts for [start, end], in UTC."""
    params = {
        "latitude": cfg.station.latitude,
        "longitude": cfg.station.longitude,
        "hourly": ",".join(HOURLY),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    return http.get_json(cfg.openmeteo.previous_runs_url, params)


def parse_daily(js) -> pd.DataFrame:
    """Hourly response -> daily rain sums per UTC day.

    Columns: precip_fc_day1, precip_fc_day2. A day only gets a value if all
    24 hours are present (min_count=24), otherwise NaN: a partial sum would
    look like "little rain" when it really means "no data".
    """
    hourly = (js or {}).get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    df = pd.DataFrame(
        {name: pd.to_numeric(pd.Series(hourly.get(name, [None] * len(times)), dtype="object"),
                             errors="coerce").to_numpy()
         for name in HOURLY},
        index=times,
    )
    daily = df.groupby(df.index.normalize()).sum(min_count=24)
    daily.index.name = "date"
    return daily.rename(columns={
        "precipitation_previous_day1": "precip_fc_day1",
        "precipitation_previous_day2": "precip_fc_day2",
    })
