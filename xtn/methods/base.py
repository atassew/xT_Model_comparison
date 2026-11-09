"""Common interface shared by all action-valuation methods."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ActionValuer(ABC):
    """An action-valuation method.

    ``fit`` learns the model from a set of games; ``rate`` returns one value
    per action (aligned to ``actions.index``). Actions that a method does not
    value get NaN; downstream aggregation treats NaN as 0.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, actions: pd.DataFrame, games: pd.DataFrame) -> "ActionValuer":
        ...

    @abstractmethod
    def rate(self, actions: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
        ...
