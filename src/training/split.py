"""Split feature rows by target date into train, validation and test.

Rows are indexed by target date d, and every feature of row d uses observations
dated <= d-2 (see common/features.py). So a split on the target date is enough:
no training label lies inside the validation window. There is no shuffling.

The test window is returned but must never be used to choose or tune a model
(model selection rule in the README).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


class LeakageError(Exception):
    """A split would let training data overlap the validation or test window."""


@dataclass(frozen=True)
class SplitDates:
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        if not self.train_start <= self.train_end < self.val_start <= self.val_end < self.test_start <= self.test_end:
            raise LeakageError(
                "Windows must be in order train < validation < test without overlap, got "
                f"train {self.train_start}..{self.train_end}, val {self.val_start}..{self.val_end}, "
                f"test {self.test_start}..{self.test_end}"
            )


@dataclass(frozen=True, eq=False)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _window(rows: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return rows[(rows.index >= pd.Timestamp(start)) & (rows.index <= pd.Timestamp(end))]


def split_by_target_date(rows: pd.DataFrame, dates: SplitDates) -> Split:
    """Cut `rows` (index = target date) into the three windows of `dates`."""
    split = Split(
        train=_window(rows, dates.train_start, dates.train_end),
        validation=_window(rows, dates.val_start, dates.val_end),
        test=_window(rows, dates.test_start, dates.test_end),
    )
    # Guards the code above, not the input: if a later edit breaks _window, fail here.
    if len(split.train) and len(split.validation) and split.train.index.max() >= split.validation.index.min():
        raise LeakageError(
            f"Training rows reach {split.train.index.max().date()}, validation starts {split.validation.index.min().date()}"
        )
    return split
