"""Feature pipeline steps. Each public function is one CLI command.

Where data lives in storage is decided in `common.store.FeatureStore` (layout in its docstring);
this module only says what to fetch, compute and save.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from common.config import Config
from common.features import build_feature_rows
from common.storage import Storage
from common.store import FeatureStore
from features import clean, cosmos, openmeteo

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(ts: datetime) -> str:
    """Timestamp safe for file names, e.g. 20261030T190501Z."""
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _feature_store(cfg: Config, store: Storage) -> FeatureStore:
    return FeatureStore(store, cfg.station.id, cfg.features.version)


# ---------------------------------------------------------------- observations
def fetch_observations(cfg: Config, store: Storage, start: date, end: date) -> int:
    """Fetch COSMOS data year by year, keep the raw responses, upsert the clean table."""
    fs = _feature_store(cfg, store)
    fetched_at = utc_now()
    frames = []
    for y0, y1 in cosmos.year_windows(start, end):
        js = cosmos.fetch(cfg, y0, y1)
        if js is not None:
            # Raw first: if parsing has a bug we can always re-parse the original.
            fs.write_raw_cosmos(y0.year, stamp(fetched_at), js)
        frames.append(cosmos.parse(js, cfg.cosmos.parameters))
        log.info("COSMOS %s..%s: %d days", y0, y1, len(frames[-1]))

    new = clean.validate(pd.concat(frames), cfg.ranges)
    fs.write_observations(clean.upsert(fs.read_observations(), new))
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
    if not _feature_store(cfg, store).has_observations():
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
    _feature_store(cfg, store).write_raw_forecast(stamp(issued_at), js)

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
    fs = _feature_store(cfg, store)
    start = start or cfg.openmeteo.archive_start
    end = end or utc_now().date()
    fetched_at = utc_now()
    frames = []
    for y0, y1 in cosmos.year_windows(start, end):
        js = openmeteo.fetch(cfg, y0, y1)
        fs.write_raw_forecast_archive(stamp(fetched_at), y0.year, js)
        frames.append(openmeteo.parse_daily(js))
    new = pd.concat(frames)
    new["fc_source"] = "archive"
    _merge_forecasts(cfg, store, new)
    log.info("Forecast backfill %s..%s: %d days with day2 value", start, end, int(new["precip_fc_day2"].notna().sum()))


def _merge_forecasts(cfg: Config, store: Storage, new: pd.DataFrame) -> None:
    fs = _feature_store(cfg, store)
    new = clean.validate(new, {"precip_fc_day1": (0, 300), "precip_fc_day2": (0, 300)})
    fs.write_forecasts(clean.upsert_forecasts(fs.read_forecasts(), new))


# ---------------------------------------------------------------- features
def build_features(cfg: Config, store: Storage, reference_date: date | None = None) -> pd.DataFrame:
    """Recompute the whole feature table from the clean tables and save it as the
    snapshot of `reference_date` (default: today, UTC).

    The table is small (~1,400 rows), so rebuilding everything each run is
    simpler and safer than incremental updates: the result only depends on
    the clean data, never on the order of earlier runs.
    """
    reference_date = reference_date or utc_now().date()
    fs = _feature_store(cfg, store)
    today = pd.Timestamp(reference_date)
    feats = build_feature_rows(
        fs.read_observations(), fs.read_forecasts(),
        issue_gap=cfg.features.issue_gap, last_target=today + pd.Timedelta(days=1),
    )

    path = fs.write_features(feats, reference_date, {
        "updated_at": utc_now().isoformat(),
        "rows": len(feats),
        "labelled_rows": int(feats["is_labelled"].sum()),
        "last_target_date": feats.index.max().date().isoformat(),
    })
    _freeze_issued_row(fs, feats, today + pd.Timedelta(days=1))
    log.info("Features: %d rows, %d labelled, last target %s, saved to %s",
             len(feats), int(feats["is_labelled"].sum()), feats.index.max().date(), path)
    return feats


def _freeze_issued_row(fs: FeatureStore, feats: pd.DataFrame, target: pd.Timestamp) -> None:
    """Keep the row for tomorrow exactly as it was on the issue day.

    Later runs may see corrected data and compute a slightly different row;
    the live evaluation (MS4) must use what was really known when the forecast
    was made. Written only once, and only when the row is complete (latest
    observation and today's forecast snapshot present); otherwise a later run
    the same day tries again.
    """
    if target not in feats.index or fs.is_issued(target.date()):
        return
    row = feats.loc[target]
    if not (row["inference_ok"] and pd.notna(row["precip_fc_gap_sum"])):
        log.info("Row for target %s not complete yet, not frozen", target.date())
        return
    fs.write_issued(target.date(), feats.loc[[target]].assign(frozen_at=utc_now().isoformat()))
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
    _feature_store(cfg, store).append_probe(json.dumps(entry))
    log.info("Probe: %s", entry)
    return entry
