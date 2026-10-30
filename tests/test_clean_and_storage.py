"""Validation, upserts, storage and an end-to-end run with mocked APIs."""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from common.storage import Storage
from features import clean, pipeline

RANGES = {"cosmos_vwc": (0, 100), "precip": (0, 300)}
FIXTURES = Path(__file__).parent / "fixtures"


def frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows], name="date")
    return pd.DataFrame([r[1] for r in rows], index=idx)


def test_validate_dedups_ranges_and_drops_empty_columns():
    df = frame([
        ("2025-01-01", {"cosmos_vwc": 30.0, "precip": 1.0, "empty": np.nan}),
        ("2025-01-02", {"cosmos_vwc": 150.0, "precip": -2.0, "empty": np.nan}),  # out of range
        ("2025-01-01", {"cosmos_vwc": 31.0, "precip": 1.0, "empty": np.nan}),  # duplicate date
    ])
    out = clean.validate(df, RANGES)
    assert list(out.index) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]
    assert out.loc["2025-01-01", "cosmos_vwc"] == 31.0  # last duplicate kept
    assert np.isnan(out.loc["2025-01-02", "cosmos_vwc"])
    assert np.isnan(out.loc["2025-01-02", "precip"])
    assert "empty" not in out.columns


def test_upsert_is_idempotent_and_newest_wins():
    old = frame([("2025-01-01", {"cosmos_vwc": 30.0}), ("2025-01-02", {"cosmos_vwc": 31.0})])
    new = frame([("2025-01-02", {"cosmos_vwc": 31.5}), ("2025-01-03", {"cosmos_vwc": 32.0})])
    once = clean.upsert(old, new)
    twice = clean.upsert(once, new)
    pd.testing.assert_frame_equal(once, twice)
    assert once.loc["2025-01-02", "cosmos_vwc"] == 31.5  # corrected value replaces old one
    assert len(once) == 3


def test_forecast_upsert_keeps_first_value():
    old = frame([("2025-01-02", {"precip_fc_day1": np.nan, "precip_fc_day2": 4.0, "fc_source": "live"})])
    new = frame([("2025-01-02", {"precip_fc_day1": 1.0, "precip_fc_day2": 9.0, "fc_source": "archive"})])
    out = clean.upsert_forecasts(old, new)
    assert out.loc["2025-01-02", "precip_fc_day2"] == 4.0  # issued value is never overwritten
    assert out.loc["2025-01-02", "precip_fc_day1"] == 1.0  # but gaps are filled
    assert out.loc["2025-01-02", "fc_source"] == "live"


def test_storage_roundtrip_local(tmp_path):
    store = Storage(str(tmp_path))
    df = frame([("2025-01-01", {"x": 1.0})])
    store.write_parquet("a/b/t.parquet", df)
    pd.testing.assert_frame_equal(store.read_parquet("a/b/t.parquet"), df, check_freq=False)
    store.write_json("p.json", {"k": 1})
    assert store.read_json("p.json") == {"k": 1}
    store.append_line("log.jsonl", '{"n": 1}')
    store.append_line("log.jsonl", '{"n": 2}')
    assert store.read_bytes("log.jsonl").decode().splitlines() == ['{"n": 1}', '{"n": 2}']
    assert store.list("a") == ["a/b/t.parquet"]
    assert not list(tmp_path.rglob("*.tmp"))  # atomic write leaves no temp files


# ------------------------------------------------------------------ end to end
@pytest.fixture
def mocked(monkeypatch, tmp_path):
    """Run the real pipeline with the APIs and the clock replaced.

    COSMOS returns the real BALRD fixture (1-10 March 2025); "now" is
    2025-03-11 10:00 UTC, so yesterday (03-10) is the latest observation.
    """
    cosmos_js = json.loads((FIXTURES / "cosmos_balrd_2025-03-01_10.json").read_text())
    state = {"cosmos": cosmos_js}

    def fake_get_json(url, params=None, **kw):
        if "cosmos" in url:
            return state["cosmos"]
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
        n = len(times)
        return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                           "precipitation_previous_day1": [0.1] * n,
                           "precipitation_previous_day2": [0.2] * n}}

    monkeypatch.setattr("common.http.get_json", fake_get_json)
    monkeypatch.setattr(pipeline, "utc_now", lambda: datetime(2025, 3, 11, 10, 0, tzinfo=timezone.utc))
    monkeypatch.setenv("STORAGE_URI", str(tmp_path))
    cfg = load_config(Path(__file__).parents[1] / "config" / "settings.yaml")
    return cfg, Storage(cfg.storage_uri), state


def test_end_to_end_backfill_is_idempotent(mocked):
    cfg, store, _ = mocked
    pipeline.backfill(cfg, store, date(2025, 3, 1), date(2025, 3, 10))
    clean1 = store.read_bytes("clean/BALRD/observations.parquet")
    feats1 = store.read_bytes("features/v1/BALRD/features.parquet")

    pipeline.backfill(cfg, store, date(2025, 3, 1), date(2025, 3, 10))
    assert store.read_bytes("clean/BALRD/observations.parquet") == clean1
    assert store.read_bytes("features/v1/BALRD/features.parquet") == feats1

    f = store.read_parquet("features/v1/BALRD/features.parquet")
    assert f.index.max() == pd.Timestamp("2025-03-12")  # today + 1
    assert f.loc["2025-03-12", "vwc_latest"] == 32.6  # obs(03-10)
    assert store.read_json("features/latest.json")["version"] == "v1"


def test_ingest_picks_up_corrections(mocked):
    cfg, store, state = mocked
    pipeline.backfill(cfg, store, date(2025, 3, 1), date(2025, 3, 10))

    corrected = json.loads(json.dumps(state["cosmos"]))
    corrected["coverages"][0]["ranges"]["cosmos_vwc"]["values"][-1] = 33.3  # provider fixes 03-10
    state["cosmos"] = corrected
    pipeline.ingest(cfg, store)

    obs = store.read_parquet("clean/BALRD/observations.parquet")
    assert obs.loc["2025-03-10", "cosmos_vwc"] == 33.3
    assert len(obs) == 10  # no duplicate rows


def test_issued_row_frozen_only_when_complete(mocked):
    cfg, store, _ = mocked
    issued = "issued/v1/BALRD/target_date=2025-03-12.parquet"
    pipeline.backfill(cfg, store, date(2025, 3, 1), date(2025, 3, 10))
    assert not store.exists(issued)  # no forecast snapshot yet -> not frozen

    pipeline.forecast_backfill(cfg, store, date(2025, 3, 1))
    pipeline.forecast_snapshot(cfg, store)
    pipeline.build_features(cfg, store)
    assert store.exists(issued)
    row = store.read_parquet(issued)
    assert row["precip_fc_gap_sum"].iloc[0] == pytest.approx(24 * 0.1 + 24 * 0.2)


def test_probe_appends_line(mocked):
    cfg, store, _ = mocked
    entry = pipeline.probe(cfg, store)
    assert entry["yesterday"] == "2025-03-10" and entry["present"] and entry["vwc"] == 32.6
    lines = store.read_bytes("logs/feed_probe.jsonl").decode().splitlines()
    assert json.loads(lines[0])["station"] == "BALRD"
