"""Karun Singh's Expected Threat (xT), implemented from scratch.

Reference: K. Singh (2018), "Introducing Expected Threat (xT)",
https://karun.in/blog/expected-threat.html

The pitch (105 x 68 m, acting team attacking left to right) is divided into
an l x w grid (16 x 12). For every zone z, estimated from the training data:

    s(z)        share of actions in z that are shots        (shot probability)
    m(z)        share of actions in z that are moves        (move probability)
    g(z)        share of shots from z that are goals        (scoring probability)
    T(z -> z')  share of moves from z that end successfully in z'
                (failed moves stay in the denominator but go nowhere)

with "actions" = shots + moves, and moves = open-play passes, crosses and
carries (Singh's moves: passes and carries both move the ball). Shots are
open-play shots only (no penalties or direct free kicks).
Every successful move is rated, carries included; only passes and crosses
count in a team's total (``config.MOVE_TYPES``, applied when summing).

    xT(z) = s(z) * g(z) + m(z) * sum_z' T(z -> z') * xT(z')

is solved by value iteration from xT = 0 until no zone changes by more than
eps. A successful move is valued xT(end zone) - xT(start zone).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import ActionValuer

# Singh's moves (fit and rating). Carries are needed for realistic transitions,
# even though only passes and crosses are summed in the study.
MOVE_TYPES = ["pass", "cross", "carry"]


class KarunXT(ActionValuer):
    name = "xT"

    def __init__(self, l: int = 16, w: int = 12, eps: float = 1e-5, max_iter: int = 1000,
                 penalise_failed: bool = False):
        """
        Parameters
        ----------
        l, w : grid size along / across the pitch (Singh uses 16 x 12).
        eps : convergence tolerance of the value iteration.
        penalise_failed : if True, a failed move gets -xT(start zone). Singh's
            original only values successful moves, so the default is False.
        """
        self.l, self.w, self.eps, self.max_iter = l, w, eps, max_iter
        self.penalise_failed = penalise_failed
        self.grid: np.ndarray | None = None  # shape (w, l): row = y band, column = x band
        self.n_iter = 0

    # ---- helpers -----------------------------------------------------------
    def _cell(self, x, y) -> np.ndarray:
        """Flat zone index (row-major over (w, l)) for coordinates in metres."""
        xi = np.clip((np.asarray(x) / config.FIELD_LENGTH * self.l).astype(int), 0, self.l - 1)
        yj = np.clip((np.asarray(y) / config.FIELD_WIDTH * self.w).astype(int), 0, self.w - 1)
        return yj * self.l + xi

    # ---- fit -------------------------------------------------------------------
    def fit(self, actions, games=None):
        n = self.l * self.w
        moves = actions[actions.type_name.isin(MOVE_TYPES)]
        shots = actions[actions.type_name == "shot"]

        move_start = self._cell(moves.start_x.values, moves.start_y.values)
        shot_start = self._cell(shots.start_x.values, shots.start_y.values)
        n_moves = np.bincount(move_start, minlength=n).astype(float)
        n_shots = np.bincount(shot_start, minlength=n).astype(float)
        n_goals = np.bincount(shot_start[(shots.result_name == "success").values], minlength=n).astype(float)

        total = n_moves + n_shots
        with np.errstate(divide="ignore", invalid="ignore"):
            self.shot_prob = np.where(total > 0, n_shots / total, 0.0)
            self.move_prob = np.where(total > 0, n_moves / total, 0.0)
            self.score_prob = np.where(n_shots > 0, n_goals / n_shots, 0.0)

        ok = (moves.result_name == "success").values
        s_ok = move_start[ok]
        e_ok = self._cell(moves.end_x.values[ok], moves.end_y.values[ok])
        counts = np.zeros((n, n))
        np.add.at(counts, (s_ok, e_ok), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition = np.where(n_moves[:, None] > 0, counts / n_moves[:, None], 0.0)

        xt = np.zeros(n)
        base = self.shot_prob * self.score_prob
        for it in range(1, self.max_iter + 1):
            new = base + self.move_prob * (self.transition @ xt)
            done = np.max(np.abs(new - xt)) < self.eps
            xt = new
            if done:
                break
        self.n_iter = it
        self.grid = xt.reshape(self.w, self.l)
        return self

    # ---- rate ------------------------------------------------------------------
    def rate(self, actions, games=None):
        flat = self.grid.ravel()
        start = flat[self._cell(actions.start_x.values, actions.start_y.values)]
        end = flat[self._cell(actions.end_x.values, actions.end_y.values)]
        is_move = actions.type_name.isin(MOVE_TYPES).values
        ok = is_move & (actions.result_name == "success").values
        v = np.full(len(actions), np.nan)
        v[ok] = end[ok] - start[ok]
        if self.penalise_failed:
            fail = is_move & ~ok
            v[fail] = -start[fail]
        return pd.Series(v, index=actions.index)
