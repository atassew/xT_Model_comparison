"""David Sumpter's action-based xT from possession chains (Soccermatics).

Reference: Soccermatics course, Lesson 4
  - "Isolating possession chains"   (plot_PossesionChain.py)
  - "Calculating xT (action-based)" (plot_ActionBasedExpectedThreat.py)
  https://github.com/soccermatics/Soccermatics/tree/main/course/lessons/lesson4

Steps, following the course code:
 1. Split the game into possession chains. The course builds its own chains
    from Wyscout events; here StatsBomb's possession counter (read directly
    from the event JSON) is used instead. As in the course, a shot always
    ends a chain: a StatsBomb possession that continues after a shot (e.g.
    a blocked shot the team wins back) is split after every shot of the
    team in possession.
    A "shot" is an open-play shot only, as in the course: there a shot is
    Wyscout's eventName "Shot", and Wyscout records penalties and direct
    free-kick shots as "Free Kick" events.
 2. A chain "ends with a shot" if the last action of the team in possession
    is a shot. Every action in the chain gets shot_end = 1 and the xG of that
    shot (StatsBomb's xG; the course uses its own simple xG model). If the chain(s) directly before it belong to the same team and were
    stopped by a foul of the opponent, they get the same label (the course's
    foul rule). A chain that ends with the team's OWN foul never gets it.
 3. Keep only moves (open-play passes and crosses, ``config.MOVE_TYPES``)
    made by the team in possession of the chain, as in the course (Wyscout's
    "Pass" events: open-play passes, crosses included). Carries are not used:
    a move's value is a level, so a short carry after a pass would count the
    pass's end point a second time.
 4. Features: x0, x1 (start/end distance along the pitch) and c0, c1
    (start/end distance from the centre line, |y - 34|), plus all products
    of these up to degree 3 (34 features in total).
 5. Model A (classifier): P(chain ends with a shot | move). The course's
    final model is an XGBoost classifier (100 trees, depth 4, XGBoost's
    default learning rate 0.3); logistic regression is an option.
 6. Model B (linear regression), trained on moves from shot-ending chains
    only: E[xG of the shot | move].
 7. Move value = P(shot) * E[xG]. This is a *level*, not a difference
    (unlike Singh's xT), exactly as in the course.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .. import config
from .base import ActionValuer

BASE_VARS = ["x0", "x1", "c0", "c1"]


def _poly_features(df: pd.DataFrame, degree: int = 3) -> pd.DataFrame:
    """All products of x0, x1, c0, c1 up to ``degree`` (as in the course)."""
    out = {}
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(BASE_VARS, d):
            name = "".join(combo)
            v = np.ones(len(df))
            for c in combo:
                v = v * df[c].values
            out[name] = v
    return pd.DataFrame(out, index=df.index)


def label_chains(actions: pd.DataFrame, shot_types=("shot",)) -> pd.DataFrame:
    """Add ``chain``, ``shot_end`` and ``chain_xg`` columns to every action.

    ``actions`` must contain possession, possession_team_id and xg.
    ``shot_types``: the actions that count as a shot (default: open-play shots,
    as in the course; penalties and direct free kicks are not shots there).
    """
    a = actions.copy()
    a["type_name"] = a.type_name.astype(str)

    is_shot = a.type_name.isin(shot_types).astype(bool)

    # chain = StatsBomb possession, split after every shot of the team in possession
    # (a shot ends a chain, as in the course). ``part`` counts the shots before an
    # action in its possession, so each shot stays in the chain it ends.
    own_shot = (is_shot & (a.team_id == a.possession_team_id)).astype(int)
    part = own_shot.groupby([a.game_id, a.possession]).cumsum() - own_shot
    a["chain"] = (a.game_id.astype("int64") * 10_000 + a.possession.astype("int64")) * 100 + part

    # every chain of the game, in order, with the team in possession
    last_any = a.groupby("chain").tail(1)
    chain_info = pd.DataFrame({
        "chain": last_any.chain.values,
        "game_id": last_any.game_id.values,
        "possession": last_any.possession.values,
        "team": last_any.possession_team_id.values,
        # stopped by a foul of the OTHER team (an attacker's own foul gives the ball away)
        "ended_foul": (last_any.type_name.eq("foul")
                       & last_any.team_id.ne(last_any.possession_team_id)).values,
    })

    # did the chain end with a shot? (last action of the chain made by the team in possession)
    own = a[a.team_id == a.possession_team_id]
    last_own = own.groupby("chain").tail(1)
    shot = pd.DataFrame({
        "chain": last_own.chain.values,
        "shot_end": is_shot.loc[last_own.index].values.astype(int),
        "chain_xg": np.where(is_shot.loc[last_own.index], last_own.xg.fillna(0), 0.0),
    })
    chain_info = chain_info.merge(shot, on="chain", how="left").fillna({"shot_end": 0, "chain_xg": 0.0})

    # course foul rule: propagate the shot label back over the chains directly
    # before it that belong to the same team and were stopped by an opponent's foul
    chain_info = chain_info.sort_values("chain").reset_index(drop=True)  # game, possession, part
    shot_end = chain_info.shot_end.values.astype(int)
    chain_xg = chain_info.chain_xg.values.copy()
    for i in np.flatnonzero(shot_end == 1):
        k = i - 1
        while (k >= 0 and chain_info.game_id[k] == chain_info.game_id[i]
               and chain_info.ended_foul[k] and chain_info.team[k] == chain_info.team[i]):
            shot_end[k] = 1
            chain_xg[k] = chain_info.chain_xg[i]
            k -= 1
    chain_info["shot_end"] = shot_end
    chain_info["chain_xg"] = chain_xg

    a = a.merge(chain_info[["chain", "shot_end", "chain_xg"]], on="chain", how="left")
    a.index = actions.index
    a[["shot_end", "chain_xg"]] = a[["shot_end", "chain_xg"]].fillna(0)
    return a


class SumpterChainXT(ActionValuer):
    name = "Sumpter"

    def __init__(self, classifier: str = "xgboost", degree: int = 3,
                 shot_types=("shot",), random_state: int = config.RANDOM_STATE):
        """
        Parameters
        ----------
        classifier : "xgboost" (the course's final model) or "logistic".
        degree : highest degree of the polynomial location features (3 in course).
        shot_types : the actions that count as a shot (course: open-play shots only).
        """
        self.classifier = classifier
        self.degree = degree
        self.shot_types = tuple(shot_types)
        self.random_state = random_state

    def _moves(self, actions, games) -> pd.DataFrame:
        """Passes and crosses of the team in possession, with chain labels and x0, x1, c0, c1."""
        # Actions are already left-to-right for the acting team (StatsBomb convention),
        # so x = distance from the team's own goal line.
        lab = label_chains(actions, shot_types=self.shot_types)
        p = lab[lab.type_name.isin(config.MOVE_TYPES) & (lab.team_id == lab.possession_team_id)].copy()
        p["x0"] = p.start_x                                   # start: distance along the pitch
        p["x1"] = p.end_x                                     # end:   distance along the pitch
        p["c0"] = (p.start_y - config.FIELD_WIDTH / 2).abs()  # start: distance from the centre line
        p["c1"] = (p.end_y - config.FIELD_WIDTH / 2).abs()    # end:   distance from the centre line
        return p

    def fit(self, actions, games):
        p = self._moves(actions, games)
        X = _poly_features(p, self.degree)      # 34 location features
        y = p.shot_end.astype(int).values       # 1 = the move's chain ended with a shot

        # Model A: P(chain ends with a shot | move), trained on all moves

        if self.classifier == "xgboost":
            # course settings; learning rate left at XGBoost's default (0.3), as in the course
            self.clf = XGBClassifier(n_estimators=100, max_depth=4,
                                     random_state=self.random_state, n_jobs=2)
        else:
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.clf.fit(X.values, y)

        # Model B: E[xG of the shot | move], trained only on moves from shot-ending chains
        shot = p.shot_end == 1
        self.reg = LinearRegression().fit(X[shot].values, p.loc[shot, "chain_xg"].values)
        return self

    def rate(self, actions, games):
        p = self._moves(actions, games)
        X = _poly_features(p, self.degree).values
        shot_prob = self.clf.predict_proba(X)[:, 1]   # model A
        xg_pred = self.reg.predict(X)                  # model B
        values = pd.Series(np.nan, index=actions.index)  # non-moves are not valued
        values.loc[p.index] = shot_prob * xg_pred         # move value = P(shot) x E[xG]
        return values
