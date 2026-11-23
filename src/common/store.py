"""Feature store repository: the only code that knows where data lives in storage.

Pipelines ask by meaning (`read_observations()`, `read_features(snapshot_date)`) and never
build a bucket path themselves, so a change to the layout touches this file only.
It receives a `Storage` adapter and imports no config; `features.pipeline._feature_store`
builds it from the config.

Layout (relative to STORAGE_URI):
    raw/cosmos/<station>/<year>/fetched_at=<ts>.json                  as returned, never modified
    raw/openmeteo/<station>/issued_at=<ts>.json                       live forecast, as issued
    raw/openmeteo/<station>/archive/fetched_at=<ts>_<year>.json       archived forecasts
    clean/<station>/observations.parquet                              one row per date
    clean/<station>/forecasts.parquet                                 one row per date, first value wins
    features/<version>/<station>/snapshot_date=<D>/features.parquet   feature table as built on day D
    features/latest.json                                              points to the newest snapshot
    issued/<version>/<station>/target_date=<d>.parquet                inference row frozen on the issue day
    logs/feed_probe.jsonl                                             arrival-time probe (throwaway)

Snapshots (editorial E1): every rebuild writes a dated copy instead of replacing one object,
so a training run can record which state it used and a later run can read the same bytes.
Several runs on the same day replace that day's snapshot, so a snapshot is final once its
day is over. Training should pin a completed day.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import date

import pandas as pd

from common.storage import Storage


class SnapshotNotFound(Exception):
    """The requested feature snapshot does not exist in storage."""


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_date: date
    path: str
    sha256: str
    features: pd.DataFrame

    def training_rows(self) -> pd.DataFrame:
        """Rows with a label and a persistence value, the only rows a model can learn from."""
        f = self.features
        return f[f["is_labelled"] & f["inference_ok"]]


class FeatureStore:
    def __init__(self, storage: Storage, station: str, version: str):
        self.storage = storage
        self.station = station
        self.version = version

    # ------------------------------------------------------------------ paths
    @property
    def _observations(self) -> str:
        return f"clean/{self.station}/observations.parquet"

    @property
    def _forecasts(self) -> str:
        return f"clean/{self.station}/forecasts.parquet"

    @property
    def _latest(self) -> str:
        return "features/latest.json"

    def _snapshot(self, snapshot_date: date) -> str:
        return f"features/{self.version}/{self.station}/snapshot_date={snapshot_date.isoformat()}/features.parquet"

    def _issued(self, target: date) -> str:
        return f"issued/{self.version}/{self.station}/target_date={target.isoformat()}.parquet"

    def _read_or_none(self, rel: str) -> pd.DataFrame | None:
        return self.storage.read_parquet(rel) if self.storage.exists(rel) else None

    # ------------------------------------------------------------------ raw
    def write_raw_cosmos(self, year: int, stamp: str, js: object) -> None:
        self.storage.write_json(f"raw/cosmos/{self.station}/{year}/fetched_at={stamp}.json", js)

    def write_raw_forecast(self, stamp: str, js: object) -> None:
        self.storage.write_json(f"raw/openmeteo/{self.station}/issued_at={stamp}.json", js)

    def write_raw_forecast_archive(self, stamp: str, year: int, js: object) -> None:
        self.storage.write_json(f"raw/openmeteo/{self.station}/archive/fetched_at={stamp}_{year}.json", js)

    # ------------------------------------------------------------------ clean
    def has_observations(self) -> bool:
        return self.storage.exists(self._observations)

    def read_observations(self) -> pd.DataFrame | None:
        return self._read_or_none(self._observations)

    def write_observations(self, df: pd.DataFrame) -> None:
        self.storage.write_parquet(self._observations, df)

    def read_forecasts(self) -> pd.DataFrame | None:
        return self._read_or_none(self._forecasts)

    def write_forecasts(self, df: pd.DataFrame) -> None:
        self.storage.write_parquet(self._forecasts, df)

    # ------------------------------------------------------------------ features
    def write_features(self, feats: pd.DataFrame, snapshot_date: date, meta: dict) -> str:
        """Write the snapshot for `snapshot_date`, then point latest.json to it. Returns the path.

        latest.json is written second, so a reader never gets a pointer to a missing file.
        """
        buf = io.BytesIO()
        feats.to_parquet(buf)
        data = buf.getvalue()
        rel = self._snapshot(snapshot_date)
        self.storage.write_bytes(rel, data)
        self.storage.write_json(self._latest, {
            "version": self.version,
            "station": self.station,
            "snapshot_date": snapshot_date.isoformat(),
            "snapshot": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            **meta,
        })
        return rel

    def latest_snapshot_date(self) -> date:
        if not self.storage.exists(self._latest):
            raise SnapshotNotFound("features/latest.json does not exist; run build-features first")
        return date.fromisoformat(self.storage.read_json(self._latest)["snapshot_date"])

    def list_snapshot_dates(self) -> list[date]:
        prefix = f"features/{self.version}/{self.station}"
        return sorted(
            date.fromisoformat(p.split("snapshot_date=", 1)[1].split("/", 1)[0])
            for p in self.storage.list(prefix)
            if "snapshot_date=" in p
        )

    def read_features(self, snapshot_date: date | None = None) -> FeatureSnapshot:
        """Read one pinned snapshot; without a date, the one latest.json points to."""
        snapshot_date = snapshot_date or self.latest_snapshot_date()
        rel = self._snapshot(snapshot_date)
        if not self.storage.exists(rel):
            raise SnapshotNotFound(f"No feature snapshot for {snapshot_date} at {rel}")
        data = self.storage.read_bytes(rel)
        return FeatureSnapshot(
            snapshot_date=snapshot_date,
            path=rel,
            sha256=hashlib.sha256(data).hexdigest(),
            features=pd.read_parquet(io.BytesIO(data)),
        )

    # ------------------------------------------------------------------ issued rows
    def is_issued(self, target: date) -> bool:
        return self.storage.exists(self._issued(target))

    def write_issued(self, target: date, row: pd.DataFrame) -> None:
        self.storage.write_parquet(self._issued(target), row)

    # ------------------------------------------------------------------ probe
    def append_probe(self, line: str) -> None:
        self.storage.append_line("logs/feed_probe.jsonl", line)
