"""Feature pipeline steps. Each public function is one CLI command.

Storage layout (relative to STORAGE_URI):
    raw/cosmos/<station>/<year>/fetched_at=<ts>.json     as returned, never modified
    raw/openmeteo/<station>/issued_at=<ts>.json          live forecast, as issued
    raw/openmeteo/<station>/archive/fetched_at=<ts>_<year>.json
    clean/<station>/observations.parquet                 one row per date (upsert)
    clean/<station>/forecasts.parquet                    one row per date (first value wins)
    features/<version>/<station>/features.parquet        the feature store
    features/latest.json                                 pointer to the current version
    issued/<version>/<station>/target_date=<d>.parquet   inference row frozen at issue time
    logs/feed_probe.jsonl                                arrival-time probe (throwaway)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from common.config import Config
from common.features import build_feature_rows
from common.storage import Storage
from features import clean, cosmos, openmeteo

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(ts: datetime) -> str:
    """Timestamp safe for file names, e.g. 20261030T190501Z."""
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _paths(cfg: Config) -> dict[str, str]:
    s, v = cfg.station.id, cfg.features.version
    return {
        "observations": f"clean/{s}/observations.parquet",
        "forecasts": f"clean/{s}/forecasts.parquet",
        "features": f"features/{v}/{s}/features.parquet",
        "latest": "features/latest.json",
        "issued": f"issued/{v}/{s}",
        "probe": "logs/feed_probe.jsonl",
    }


def _read_or_none(store: Storage, rel: str) -> pd.DataFrame | None:
    return store.read_parquet(rel) if store.exists(rel) else None


# ---------------------------------------------------------------- observations
def fetch_observations(cfg: Config, store: Storage, start: date, end: date) -> int:
    """Fetch COSMOS data year by year, keep the raw responses, upsert the clean table."""
    fetched_at = utc_now()
    frames = []
    for y0, y1 in cosmos.year_windows(start, end):
        js = cosmos.fetch(cfg, y0, y1)
        if js is not None:
            # Raw first: if parsing has a bug we can always re-parse the original.
            store.write_json(
                f"raw/cosmos/{cfg.station.id}/{y0.year}/fetched_at={stamp(fetched_at)}.json", js
            )
        frames.append(cosmos.parse(js, cfg.cosmos.parameters))
        log.info("COSMOS %s..%s: %d days", y0, y1, len(frames[-1]))

    new = clean.validate(pd.concat(frames), cfg.ranges)
    p = _paths(cfg)["observations"]
    store.write_parquet(p, clean.upsert(_read_or_none(store, p), new))
    return len(new)


def backfill(cfg: Config, store: Storage, start: date | None = None, end: date | None = None) -> None:
    start = start or cfg.cosmos.backfill_start
    end = end or utc_now().date() - timedelta(days=1)
    n = fetch_observations(cfg, store, start, end)
    log.info("Backfill %s..%s: %d days fetched", start, end, n)
    build_features(cfg, store)


def ingest(cfg: Config, store: Storage) -> None:
    """Daily run: re-fetch the last `refetch_days` days, then rebuild features.

    Re-fetching a whole week (not just yesterday) picks up provider
    corrections and retries days that were missing on earlier runs.
    Safe to run several times a day: the upsert makes it idempotent.
    """
    p = _paths(cfg)["observations"]
    if not store.exists(p):
        # Fresh storage (e.g. new bucket): nothing to update, so do a full backfill.
        log.info("No clean table yet, running backfill first")
        return backfill(cfg, store)
    today = utc_now().date()
    fetch_observations(cfg, store, today - timedelta(days=cfg.cosmos.refetch_days), today)
    build_features(cfg, store)


# ---------------------------------------------------------------- forecasts
def forecast_snapshot(cfg: Config, store: Storage) -> None:
    """Store today's forecast exactly as issued (cannot be recovered later).

    On issue day D we keep: the forecast for D made one day earlier (day1) and
    the forecast for D+1 made two days earlier (day2). These are the two
    values `build_feature_rows` needs for target D+1.
    """
    issued_at = utc_now()
    d = issued_at.date()
    js = openmeteo.fetch(cfg, d, d + timedelta(days=1))
    store.write_json(f"raw/openmeteo/{cfg.station.id}/issued_at={stamp(issued_at)}.json", js)

    daily = openmeteo.parse_daily(js)
    new = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(d), pd.Timestamp(d + timedelta(days=1))], name="date"))
    new["precip_fc_day1"] = [daily["precip_fc_day1"].get(pd.Timestamp(d)), float("nan")]
    new["precip_fc_day2"] = [float("nan"), daily["precip_fc_day2"].get(pd.Timestamp(d + timedelta(days=1)))]
    new["fc_source"] = "live"
    _merge_forecasts(cfg, store, new)
    log.info("Forecast snapshot %s: %s", stamp(issued_at), new.to_dict("index"))


def forecast_backfill(cfg: Config, store: Storage, start: date | None = None, end: date | None = None) -> None:
    """Archived lead-time forecasts for past dates, up to today.

    Up to today (not yesterday) so there is no hole on the day the live
    snapshot starts: both values for today were made yesterday or earlier.
    """
    start = start or cfg.openmeteo.archive_start
    end = end or utc_now().date()
    fetched_at = utc_now()
    frames = []
    for y0, y1 in cosmos.year_windows(start, end):
        js = openmeteo.fetch(cfg, y0, y1)
        store.write_json(
            f"raw/openmeteo/{cfg.station.id}/archive/fetched_at={stamp(fetched_at)}_{y0.year}.json", js
        )
        frames.append(openmeteo.parse_daily(js))
    new = pd.concat(frames)
    new["fc_source"] = "archive"
    _merge_forecasts(cfg, store, new)
    log.info("Forecast backfill %s..%s: %d days with day2 value", start, end, int(new["precip_fc_day2"].notna().sum()))


def _merge_forecasts(cfg: Config, store: Storage, new: pd.DataFrame) -> None:
    new = clean.validate(new, {"precip_fc_day1": (0, 300), "precip_fc_day2": (0, 300)})
    p = _paths(cfg)["forecasts"]
    store.write_parquet(p, clean.upsert_forecasts(_read_or_none(store, p), new))


# ---------------------------------------------------------------- features
def build_features(cfg: Config, store: Storage) -> pd.DataFrame:
    """Recompute the whole feature table from the clean tables.

    The table is small (~1,400 rows), so rebuilding everything each run is
    simpler and safer than incremental updates: the result only depends on
    the clean data, never on the order of earlier runs.
    """
    p = _paths(cfg)
    obs = store.read_parquet(p["observations"])
    fc = _read_or_none(store, p["forecasts"])
    today = pd.Timestamp(utc_now().date())
    feats = build_feature_rows(obs, fc, issue_gap=cfg.features.issue_gap, last_target=today + pd.Timedelta(days=1))

    store.write_parquet(p["features"], feats)
    store.write_json(p["latest"], {
        "version": cfg.features.version,
        "station": cfg.station.id,
        "updated_at": utc_now().isoformat(),
        "rows": len(feats),
        "labelled_rows": int(feats["is_labelled"].sum()),
        "last_target_date": feats.index.max().date().isoformat(),
    })
    _freeze_issued_row(cfg, store, feats, today + pd.Timedelta(days=1))
    log.info("Features: %d rows, %d labelled, last target %s",
             len(feats), int(feats["is_labelled"].sum()), feats.index.max().date())
    return feats


def _freeze_issued_row(cfg: Config, store: Storage, feats: pd.DataFrame, target: pd.Timestamp) -> None:
    """Keep the row for tomorrow exactly as it was on the issue day.

    Later runs may see corrected data and compute a slightly different row;
    the live evaluation (MS4) must use what was really known when the forecast
    was made. Written only once, and only when the row is complete (latest
    observation and today's forecast snapshot present); otherwise a later run
    the same day tries again.
    """
    rel = f"{_paths(cfg)['issued']}/target_date={target.date().isoformat()}.parquet"
    if target not in feats.index or store.exists(rel):
        return
    row = feats.loc[target]
    if not (row["inference_ok"] and pd.notna(row["precip_fc_gap_sum"])):
        log.info("Row for target %s not complete yet, not frozen", target.date())
        return
    store.write_parquet(rel, feats.loc[[target]].assign(frozen_at=utc_now().isoformat()))
    log.info("Froze inference row for target %s", target.date())


# ---------------------------------------------------------------- probe
def probe(cfg: Config, store: Storage) -> dict:
    """Is yesterday's COSMOS record already published? Append the answer to a log.

    Run hourly for a few days to learn when the daily record arrives
    (editorial point E1), then set the ingest cron after that time and
    delete this job.
    """
    now = utc_now()
    yesterday = now.date() - timedelta(days=1)
    df = cosmos.parse(cosmos.fetch(cfg, yesterday, yesterday), ["cosmos_vwc"])
    vwc = df["cosmos_vwc"].get(pd.Timestamp(yesterday))
    present = vwc is not None and pd.notna(vwc)
    entry = {
        "checked_at": now.isoformat(),
        "station": cfg.station.id,
        "yesterday": yesterday.isoformat(),
        "present": bool(present),
        "vwc": float(vwc) if present else None,
    }
    store.append_line(_paths(cfg)["probe"], json.dumps(entry))
    log.info("Probe: %s", entry)
    return entry
