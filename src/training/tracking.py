"""MLflow adapter: experiment tracking and the model registry, behind a few named methods.

Where MLflow writes comes from the environment: MLFLOW_TRACKING_URI (+ username and token)
points to DagsHub; without it MLflow uses a local mlflow.db. The pipeline never calls
mlflow directly, so switching to W&B or another registry touches this file only.

Registry convention: one registered model per station; the version the inference
pipeline must load has the alias "champion".
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict

import lightgbm as lgb
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from training.metrics import Scores
from training.policy import ChampionDecision

CHAMPION = "champion"


class Tracker:
    def __init__(self, experiment: str, model_name: str):
        mlflow.set_experiment(experiment)
        self.model_name = model_name
        self.client = MlflowClient()

    @contextmanager
    def run(self, name: str, nested: bool = False) -> Generator[None]:
        with mlflow.start_run(run_name=name, nested=nested):
            yield

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_scores(self, prefix: str, scores: Scores) -> None:
        mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in asdict(scores).items() if v is not None})

    def log_model(self, model: lgb.Booster, features: list[str]) -> str:
        """Store the model as an artifact of the active run; returns its URI for registering."""
        mlflow.log_dict({"features": features}, "features.json")
        return mlflow.lightgbm.log_model(model, name="model").model_uri

    def log_decision(self, decision: ChampionDecision) -> None:
        mlflow.set_tags({"decision.promote": decision.promote,
                         "decision.candidate": decision.candidate or "",
                         "decision.reason": decision.reason})

    def champion(self) -> tuple[str, lgb.Booster] | None:
        """(variant name, model) of the current champion, or None if there is none yet."""
        try:
            version = self.client.get_model_version_by_alias(self.model_name, CHAMPION)
        except MlflowException:
            return None
        model = mlflow.lightgbm.load_model(f"models:/{self.model_name}@{CHAMPION}")
        return version.tags.get("variant", "unknown"), model

    def promote(self, model_uri: str, tags: dict[str, str]) -> str:
        """Register the model and move the champion alias to it. Returns the new version."""
        version = mlflow.register_model(model_uri, self.model_name, tags=tags)
        self.client.set_registered_model_alias(self.model_name, CHAMPION, version.version)
        return version.version
