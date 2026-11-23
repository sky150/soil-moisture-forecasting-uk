"""Fit and predict one variant. The only training module that imports LightGBM."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from common.features import LABEL
from training.variants import Variant


def fit(variant: Variant, train: pd.DataFrame, params: dict) -> lgb.Booster | None:
    """Train a LightGBM model; persistence has nothing to train and returns None."""
    if variant.kind == "persistence":
        return None
    params = dict(params)
    rounds = int(params.pop("num_boost_round"))
    data = lgb.Dataset(train[list(variant.features)], label=train[LABEL])
    return lgb.train(params, data, num_boost_round=rounds)


def predict(variant: Variant, model: lgb.Booster | None, rows: pd.DataFrame) -> pd.Series:
    if variant.kind == "persistence":
        return rows["vwc_latest"]
    return predict_booster(model, rows)


def predict_booster(model: lgb.Booster, rows: pd.DataFrame) -> pd.Series:
    """Predict with the columns the model was trained on, in that order."""
    return pd.Series(model.predict(rows[model.feature_name()]), index=rows.index)
