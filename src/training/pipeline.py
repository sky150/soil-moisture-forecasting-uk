"""Training pipeline: pinned snapshot -> train every variant -> score on validation -> maybe promote.

One parent MLflow run per training, one nested run per variant. The parent records the
snapshot (date, path, sha256), the code version and the split, so the run can be repeated
on the same bytes (editorial E1). The champion is chosen on the validation window only.
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import date, datetime, timezone

from common.config import Config
from common.features import LABEL
from common.store import FeatureStore, SnapshotNotFound
from training import models
from training.metrics import rain_in_gap, score
from training.policy import CandidateResult, ChampionDecision, ChampionPolicy
from training.split import SplitDates, split_by_target_date
from training.tracking import Tracker
from training.variants import VARIANTS

log = logging.getLogger(__name__)


def git_sha() -> str:
    """Commit of the code that trains. Actions sets GITHUB_SHA; the Docker image has no .git."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def last_completed_snapshot(fs: FeatureStore, today: date) -> date:
    """Newest snapshot of a day that is over. Today's can still change (several runs a day)."""
    done = [d for d in fs.list_snapshot_dates() if d < today]
    if not done:
        raise SnapshotNotFound(f"No snapshot before {today}; pass --snapshot YYYY-MM-DD to pin one")
    return done[-1]


def train(cfg: Config, fs: FeatureStore, snapshot_date: date | None = None,
          reference_date: date | None = None) -> ChampionDecision:
    today = reference_date or datetime.now(timezone.utc).date()
    snap = fs.read_features(snapshot_date or last_completed_snapshot(fs, today))
    if "precip_gap_obs" not in snap.features:
        raise SnapshotNotFound(f"{snap.path} was built before precip_gap_obs existed; run build-features")

    t = cfg.training
    split = split_by_target_date(snap.training_rows(), SplitDates(
        t.train_start, t.train_end, t.val_start, t.val_end, t.test_start, t.test_end))
    val, test = split.validation, split.test
    rainy_val = rain_in_gap(val["precip_gap_obs"])
    tracker = Tracker(t.experiment, t.model_name)
    log.info("Snapshot %s: %d train, %d validation, %d test rows",
             snap.snapshot_date, len(split.train), len(val), len(test))

    with tracker.run(f"train-{snap.snapshot_date}"):
        tracker.log_params({
            "snapshot_date": snap.snapshot_date.isoformat(), "snapshot_path": snap.path,
            "snapshot_sha256": snap.sha256, "feature_version": fs.version, "station": fs.station,
            "git_sha": git_sha(), **{k: str(v) for k, v in t.model_dump(exclude={"lightgbm"}).items()},
        })

        results, fitted, uris = [], {}, {}
        for v in VARIANTS:
            rows = v.train_rows(split.train)
            with tracker.run(v.name, nested=True):
                model = models.fit(v, rows, t.lightgbm)
                scores = score(val[LABEL], models.predict(v, model, val), rainy_val)
                tracker.log_params({"variant": v.name, "n_features": len(v.features), "n_train": len(rows),
                                    "train_first": rows.index.min().date().isoformat(),
                                    **({} if model is None else t.lightgbm)})
                tracker.log_scores("val", scores)
                if model is not None:
                    uris[v.name] = tracker.log_model(model, list(v.features))
            fitted[v.name] = (v, model)
            results.append(CandidateResult(v.name, scores))
            log.info("%-13s train from %s (n=%d): val MAE %.4f, rainy-gap MAE %s (n=%d)", v.name,
                     rows.index.min().date(), len(rows), scores.mae,
                     "-" if scores.mae_rain_gap is None else f"{scores.mae_rain_gap:.4f}", scores.n_rain_gap)

        current = None
        if (champ := tracker.champion()) is not None:
            name, model = champ
            current = CandidateResult(f"champion ({name})",
                                      score(val[LABEL], models.predict_booster(model, val), rainy_val))
            tracker.log_scores("current_champion_val", current.validation)

        decision = ChampionPolicy().decide(results, current)
        tracker.log_decision(decision)
        log.info("Decision: %s", decision.reason)

        if decision.promote:
            v, model = fitted[decision.candidate]
            # Test window: reported for the chosen model only, after the decision (never used to choose).
            test_scores = score(test[LABEL], models.predict(v, model, test), rain_in_gap(test["precip_gap_obs"]))
            tracker.log_scores("champion_test", test_scores)
            version = tracker.promote(uris[v.name], {
                "variant": v.name, "snapshot_date": snap.snapshot_date.isoformat(),
                "snapshot_sha256": snap.sha256, "val_mae": f"{decision_mae(results, v.name):.4f}",
            })
            log.info("Registered %s version %s as champion (test MAE %.4f)", t.model_name, version, test_scores.mae)
    return decision


def decision_mae(results: list[CandidateResult], name: str) -> float:
    return next(r.validation.mae for r in results if r.name == name)
