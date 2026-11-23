"""Decide which model becomes the champion (the one the inference pipeline loads).

Rule, all on the validation window only (never the test window):
1. The best candidate is the non-baseline variant with the lowest validation MAE.
2. It must beat persistence. A model that cannot beat "tomorrow = the latest value"
   adds nothing, so nothing is promoted.
3. It must beat the current champion, scored on the same validation rows. A tie keeps
   the current champion, so a retrain on unchanged data does not swap models.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from training.metrics import Scores


class MissingBaseline(Exception):
    """The persistence result is missing, so no candidate can be judged against it."""


@dataclass(frozen=True)
class CandidateResult:
    name: str
    validation: Scores


@dataclass(frozen=True)
class ChampionDecision:
    promote: bool
    candidate: str | None  # the best non-baseline candidate, promoted or not
    reason: str


class ChampionPolicy:
    def __init__(self, baseline: str = "persistence"):
        self.baseline = baseline

    def decide(self, results: Sequence[CandidateResult], current: CandidateResult | None = None) -> ChampionDecision:
        by_name = {r.name: r for r in results}
        if self.baseline not in by_name:
            raise MissingBaseline(f"No '{self.baseline}' result among {sorted(by_name)}")
        base = by_name[self.baseline].validation.mae
        models = [r for r in results if r.name != self.baseline]
        if not models:
            return ChampionDecision(False, None, "no candidate besides the baseline")

        best = min(models, key=lambda r: r.validation.mae)
        mae = best.validation.mae
        if mae >= base:
            return ChampionDecision(False, best.name, f"{best.name} MAE {mae:.4f} does not beat persistence {base:.4f}")
        if current is not None and mae >= current.validation.mae:
            return ChampionDecision(
                False, best.name,
                f"{best.name} MAE {mae:.4f} does not beat current champion {current.name} {current.validation.mae:.4f}",
            )
        return ChampionDecision(True, best.name, f"{best.name} MAE {mae:.4f} beats persistence {base:.4f}"
                                + ("" if current is None else f" and current champion {current.validation.mae:.4f}"))
