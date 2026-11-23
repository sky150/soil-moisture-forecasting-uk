"""The candidate models compared on validation (editorial B1).

- persistence:  predicts vwc_latest = obs(d-2). The baseline every model must beat.
- lgbm_no_fc:   LightGBM without the forecast-rain features, trained on all training rows.
- lgbm_with_fc: LightGBM with all features, trained only on rows that have a forecast
                (archive data starts 2024-02-05).

Neither LightGBM model sees a missing forecast in training: lgbm_no_fc never sees the
columns, lgbm_with_fc drops the rows without them. So "forecast missing" can no longer
be a pattern that exists only in training. The rows are chosen from the data, not by a
hard-coded date, so the rule still holds if the archive start changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from common.features import FEATURE_COLUMNS

FORECAST_COLUMNS: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c.startswith("precip_fc_"))


@dataclass(frozen=True)
class Variant:
    name: str
    kind: Literal["persistence", "lightgbm"]
    features: tuple[str, ...]
    needs_forecast: bool

    def train_rows(self, train: pd.DataFrame) -> pd.DataFrame:
        """The training rows this variant learns from."""
        if self.needs_forecast:
            return train[train[list(FORECAST_COLUMNS)].notna().all(axis=1)]
        return train


PERSISTENCE = Variant("persistence", "persistence", ("vwc_latest",), needs_forecast=False)
LGBM_NO_FC = Variant(
    "lgbm_no_fc", "lightgbm", tuple(c for c in FEATURE_COLUMNS if c not in FORECAST_COLUMNS), needs_forecast=False
)
LGBM_WITH_FC = Variant("lgbm_with_fc", "lightgbm", tuple(FEATURE_COLUMNS), needs_forecast=True)

VARIANTS: tuple[Variant, ...] = (PERSISTENCE, LGBM_NO_FC, LGBM_WITH_FC)
