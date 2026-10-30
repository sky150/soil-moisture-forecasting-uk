"""COSMOS-UK API client: fetch daily observations and parse CoverageJSON.

API: https://cosmos-api.ceh.ac.uk (no key, Open Government Licence).
Example request:
    /collections/1D/locations/BALRD?datetime=2023-01-01T00:00:00Z/2023-12-31T00:00:00Z
        &parameter-name=cosmos_vwc,precip,ta,rh,stp_tsoil20
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from common import http
from common.config import Config


def fetch(cfg: Config, start: date, end: date):
    """One request for [start, end]. Returns the raw JSON, or None if no data."""
    url = f"{cfg.cosmos.base_url}/collections/{cfg.cosmos.collection}/locations/{cfg.station.id}"
    params = {
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T00:00:00Z",
        "parameter-name": ",".join(cfg.cosmos.parameters),
    }
    return http.get_json(url, params)


def year_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into calendar-year pieces (one request per year keeps
    responses small and matches what the proposal planned)."""
    return [
        (max(start, date(y, 1, 1)), min(end, date(y, 12, 31)))
        for y in range(start.year, end.year + 1)
    ]


def parse(js, parameters: list[str]) -> pd.DataFrame:
    """CoverageJSON -> DataFrame with one row per calendar date.

    The response is a CoverageCollection (sometimes a single Coverage).
    `domain.axes.t.values` holds one timestamp per day and every
    `ranges[param].values` list is aligned with it BY POSITION. Values can be
    null, and parameters with no data may be absent: both end up as NaN.
    """
    empty = pd.DataFrame(columns=parameters, index=pd.DatetimeIndex([], name="date"), dtype=float)
    if js is None:
        return empty
    coverages = js.get("coverages", []) if js.get("type") == "CoverageCollection" else [js]

    frames = []
    for cov in coverages:
        times = cov["domain"]["axes"]["t"]["values"]
        # "2023-01-01T00:00:00Z" stands for that calendar day; keep only the date.
        dates = pd.to_datetime(times, utc=True).tz_localize(None).normalize()
        data = {}
        for name in parameters:
            values = cov.get("ranges", {}).get(name, {}).get("values")
            if values is None or len(values) != len(dates):
                continue  # absent parameter -> added as NaN below
            data[name] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy()
        frames.append(pd.DataFrame(data, index=dates))

    if not frames:
        return empty
    df = pd.concat(frames)
    df.index.name = "date"
    return df.reindex(columns=parameters).astype(float)
