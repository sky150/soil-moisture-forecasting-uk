"""Scores for one model on one window: MAE overall and on days with rain in the gap.

"Rain in the gap" (editorial B1): at least RAIN_GAP_MM of observed rain on d-1 and d
together, the two days nobody has observed at issue time. These are the days where
the forecast-rain features should help most. Rows with unknown observed rain are not
counted as rainy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RAIN_GAP_MM = 1.0


@dataclass(frozen=True)
class Scores:
    n: int
    mae: float
    n_rain_gap: int
    mae_rain_gap: float | None  # None when the window has no rainy gap day


def rain_in_gap(precip_gap_obs: pd.Series, threshold_mm: float = RAIN_GAP_MM) -> pd.Series:
    return precip_gap_obs.fillna(0.0) >= threshold_mm


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def score(y_true: pd.Series, y_pred: pd.Series, rainy: pd.Series) -> Scores:
    """All three Series share one index (target dates)."""
    rainy = rainy.reindex(y_true.index, fill_value=False).to_numpy()
    n_rain = int(rainy.sum())
    return Scores(
        n=len(y_true),
        mae=mae(y_true, y_pred),
        n_rain_gap=n_rain,
        mae_rain_gap=mae(y_true[rainy], y_pred[rainy]) if n_rain else None,
    )
