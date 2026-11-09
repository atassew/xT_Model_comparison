"""VAEP - Valuing Actions by Estimating Probabilities, implemented from scratch.

Reference: T. Decroos, L. Bransen, J. Van Haaren, J. Davis (2019),
"Actions Speak Louder than Goals: Valuing Player Actions in Soccer", KDD.

Game state
    S_i = (a_i, a_{i-1}, a_{i-2}). The previous actions are expressed from the
    point of view of the team performing a_i (an opponent's action is mirrored).
    At the start of a game, missing previous actions are filled with a_i.

Labels
    scores_i   = 1 if the team performing a_i scores within actions i..i+9
    concedes_i = 1 if that team concedes within actions i..i+9
    (goals = successful shots; an own goal counts for the other team)

Models
    Two gradient-boosted tree classifiers (XGBoost) estimate
    P_scores(S_i) and P_concedes(S_i). Settings as in socceraction's default:
    100 trees, depth 3, XGBoost's default learning rate (0.3), early stopping
    after 10 rounds without AUC gain on a random 25% of the training actions
    (which are then not used for fitting).

Value (as in the paper and its reference implementation)
    prev_scores   = P_scores(S_{i-1})   if a_{i-1} is by the same team, else P_concedes(S_{i-1})
    prev_concedes = P_concedes(S_{i-1}) if a_{i-1} is by the same team, else P_scores(S_{i-1})
    both are set to 0 for the first action of a game or period, after a goal,
    and when more than 10 seconds passed since a_{i-1}. As in the reference
    implementation, prev_scores is fixed at 0.792453 before a penalty and at
    0.0465 before a corner.

    offensive_i = P_scores(S_i) - prev_scores
    defensive_i = -(P_concedes(S_i) - prev_concedes)
    VAEP(a_i)   = offensive_i + defensive_i

Features (for each of a_i, a_{i-1}, a_{i-2})
    one-hot action type, result, type x result and body part; start/end x and
    y; distance and angle to the opponent goal at start and end; dx, dy,
    movement; period; time in period and in the game.
Game-state features
    whether a_{i-1}, a_{i-2} are by the same team; time since a_{i-1}, a_{i-2};
    dx, dy and distance from the end of a_{i-1}, a_{i-2} to the start of a_i;
    goals scored, conceded and goal difference before a_i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .. import config
from ..actions import ACTION_TYPES, BODYPARTS, RESULTS
from .base import ActionValuer

GOAL_X, GOAL_Y = config.FIELD_LENGTH, config.FIELD_WIDTH / 2
SHOTS = ["shot", "shot_freekick", "shot_penalty"]
N_PREV = 3
HORIZON = 10

# socceraction's fixed P_scores before a penalty / a corner (formula.py)
PENALTY_PREV_SCORES = 0.792453
CORNER_PREV_SCORES = 0.046500

# (type, result) pairs that xtn/actions.py can produce: the one-hot "type x result"
# features (socceraction's actiontype_result_onehot). Pairs that never occur would
# be all-zero columns, which a tree never splits on, so they are left out.
_PASS_LIKE = ["pass", "cross", "throw_in", "freekick_crossed", "freekick_short",
              "corner_crossed", "corner_short", "goalkick"]
TYPE_RESULTS = (
    [(t, r) for t in _PASS_LIKE for r in ["fail", "success", "offside"]]
    + [("carry", "success"), ("clearance", "success"), ("foul", "fail"),
       ("bad_touch", "fail"), ("own_goal", "owngoal")]
    + [(t, r) for t in ["take_on", "shot", "shot_freekick", "shot_penalty", "tackle",
                        "interception", "keeper_save", "keeper_claim", "keeper_punch"]
       for r in ["fail", "success"]]
)


def _goal_team(actions: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    """Team credited with a goal on each action (NaN if the action is no goal)."""
    g = games.set_index("game_id")
    home = actions.game_id.map(g.home_team_id).values
    away = actions.game_id.map(g.away_team_id).values
    team = actions.team_id.values.astype(float)
    is_goal = actions.type_name.isin(SHOTS).values & (actions.result_name == "success").values
    is_og = (actions.type_name == "own_goal").values
    other = np.where(team == home, away, home).astype(float)
    out = np.full(len(actions), np.nan)
    out[is_goal] = team[is_goal]
    out[is_og] = other[is_og]
    return out


def _shift(arr: np.ndarray, k: int, game: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr shifted down by k within each game; returns (values, valid mask)."""
    if k == 0:
        return arr, np.ones(len(arr), bool)
    out = np.empty_like(arr)
    out[k:] = arr[:-k]
    out[:k] = arr[:k]
    valid = np.zeros(len(arr), bool)
    valid[k:] = game[k:] == game[:-k]
    return out, valid


def compute_features(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """VAEP game-state features. ``actions`` must be sorted by game and action order."""
    a = actions
    game = a.game_id.values
    team = a.team_id.values
    t_game = a.time_seconds.values.astype(float) + (a.period_id.values - 1) * 45 * 60
    base = {
        "sx": a.start_x.values.astype(float), "sy": a.start_y.values.astype(float),
        "ex": a.end_x.values.astype(float), "ey": a.end_y.values.astype(float),
        "type": pd.Categorical(a.type_name, categories=ACTION_TYPES).codes,
        "result": pd.Categorical(a.result_name, categories=RESULTS).codes,
        "body": pd.Categorical(a.bodypart_name, categories=BODYPARTS).codes,
        "period": a.period_id.values.astype(float),
        "t_period": a.time_seconds.values.astype(float),
        "t_game": t_game,
    }
    F = {}
    prev = {}
    for j in range(N_PREV):
        sh = {k: _shift(v, j, game) for k, v in base.items()}
        valid = sh["sx"][1]
        v = {k: np.where(valid, s[0], base[k]) for k, s in sh.items()}  # fill with a_i
        tm, _ = _shift(team, j, game)
        same = np.where(valid, tm == team, True)
        # express previous actions from the point of view of a_i's team
        for c, L in [("sx", config.FIELD_LENGTH), ("ex", config.FIELD_LENGTH),
                     ("sy", config.FIELD_WIDTH), ("ey", config.FIELD_WIDTH)]:
            v[c] = np.where(same, v[c], L - v[c])
        p = f"a{j}_"
        for i, name in enumerate(ACTION_TYPES):
            F[p + "type_" + name] = (v["type"] == i)
        for i, name in enumerate(RESULTS):
            F[p + "result_" + name] = (v["result"] == i)
        for t, r in TYPE_RESULTS:
            F[p + "type_result_" + t + "_" + r] = (v["type"] == ACTION_TYPES.index(t)) & \
                                                   (v["result"] == RESULTS.index(r))
        for i, name in enumerate(BODYPARTS):
            F[p + "body_" + name] = (v["body"] == i)
        F[p + "start_x"], F[p + "start_y"] = v["sx"], v["sy"]
        F[p + "end_x"], F[p + "end_y"] = v["ex"], v["ey"]
        for s, (x, y) in {"start": (v["sx"], v["sy"]), "end": (v["ex"], v["ey"])}.items():
            dx, dy = GOAL_X - x, np.abs(GOAL_Y - y)
            F[p + s + "_dist_goal"] = np.hypot(dx, dy)
            F[p + s + "_angle_goal"] = np.arctan2(dy, dx)
        F[p + "dx"] = v["ex"] - v["sx"]
        F[p + "dy"] = v["ey"] - v["sy"]
        F[p + "movement"] = np.hypot(F[p + "dx"], F[p + "dy"])
        F[p + "period"] = v["period"]
        F[p + "time_period"] = v["t_period"]
        F[p + "time_game"] = v["t_game"]
        if j > 0:
            F[p + "same_team"] = same
        prev[j] = v

    for j in range(1, N_PREV):
        F[f"time_delta_{j}"] = prev[0]["t_game"] - prev[j]["t_game"]
        # from the end of a_{i-j} to the start of a_i (socceraction's space_delta)
        F[f"space_dx_{j}"] = prev[j]["ex"] - prev[0]["sx"]
        F[f"space_dy_{j}"] = prev[j]["ey"] - prev[0]["sy"]
        F[f"space_delta_{j}"] = np.hypot(F[f"space_dx_{j}"], F[f"space_dy_{j}"])

    # score before a_i, from a_i's team's point of view
    gt = _goal_team(a, games)
    total_goals = pd.Series(~np.isnan(gt), index=a.index).groupby(game).cumsum().values - (~np.isnan(gt))
    own_goals = np.zeros(len(a))
    for tid in np.unique(team):
        is_t = (gt == tid)
        cum = pd.Series(is_t, index=a.index).groupby(game).cumsum().values - is_t
        own_goals = np.where(team == tid, cum, own_goals)
    F["goals_scored"] = own_goals
    F["goals_conceded"] = total_goals - own_goals
    F["goal_diff"] = own_goals - (total_goals - own_goals)
    X = pd.DataFrame(F, index=a.index)
    return X.astype("float32")


def compute_labels(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    game = actions.game_id.values
    team = actions.team_id.values.astype(float)
    gt = _goal_team(actions, games)
    n = len(actions)
    scores = np.zeros(n, bool)
    concedes = np.zeros(n, bool)
    for k in range(HORIZON):
        fut = np.full(n, np.nan)
        same_game = np.zeros(n, bool)
        if k == 0:
            fut, same_game = gt, np.ones(n, bool)
        else:
            fut[:-k] = gt[k:]
            same_game[:-k] = game[:-k] == game[k:]
        has = same_game & ~np.isnan(fut)
        scores |= has & (fut == team)
        concedes |= has & (fut != team)
    return pd.DataFrame({"scores": scores, "concedes": concedes}, index=actions.index)


def vaep_formula(actions: pd.DataFrame, p_scores: np.ndarray, p_concedes: np.ndarray) -> pd.DataFrame:
    game = actions.game_id.values
    team = actions.team_id.values
    t = actions.time_seconds.values.astype(float)
    period = actions.period_id.values
    prev_s, valid = _shift(p_scores, 1, game)
    prev_c, _ = _shift(p_concedes, 1, game)
    prev_team, _ = _shift(team, 1, game)
    prev_t, _ = _shift(t, 1, game)
    prev_period, _ = _shift(period, 1, game)
    is_goal = (actions.type_name.isin(SHOTS) & (actions.result_name == "success")).values | \
              (actions.type_name == "own_goal").values
    prev_goal, _ = _shift(is_goal, 1, game)

    same = prev_team == team
    ps = np.where(same, prev_s, prev_c)
    pc = np.where(same, prev_c, prev_s)
    reset = ~valid | (prev_period != period) | prev_goal | (np.abs(t - prev_t) > 10)
    ps[reset] = 0.0
    pc[reset] = 0.0
    # fixed odds of scoring before a penalty or a corner (offensive side only, as in socceraction)
    ps[(actions.type_name == "shot_penalty").values] = PENALTY_PREV_SCORES
    ps[actions.type_name.isin(["corner_crossed", "corner_short"]).values] = CORNER_PREV_SCORES
    off = p_scores - ps
    dfn = -(p_concedes - pc)
    return pd.DataFrame({"offensive_value": off, "defensive_value": dfn, "vaep_value": off + dfn},
                        index=actions.index)


class VAEPValuer(ActionValuer):
    name = "VAEP"

    def __init__(self, max_train_games: int | None = None, chunk_games: int = 150,
                 tree_params: dict | None = None, val_size: float = 0.25,
                 random_state: int = config.RANDOM_STATE):
        """
        Parameters
        ----------
        max_train_games : random subsample of training games (None = all).
        chunk_games : games per chunk when computing features (memory).
        tree_params : XGBoost parameters (default: socceraction's, 100 trees, depth 3,
            XGBoost's default learning rate 0.3, early stopping after 10 rounds on AUC).
        val_size : share of the training games held out for early stopping.
        """
        self.max_train_games = max_train_games
        self.chunk_games = chunk_games
        self.tree_params = tree_params or dict(n_estimators=100, max_depth=3,
                                               eval_metric="auc", early_stopping_rounds=10)
        self.val_size = val_size
        self.random_state = random_state
        self.last_components: pd.DataFrame | None = None

    def _features(self, actions, games):
        ids = actions.game_id.unique()
        parts = []
        for i in range(0, len(ids), self.chunk_games):
            sub = actions[actions.game_id.isin(ids[i:i + self.chunk_games])]
            parts.append(compute_features(sub, games))
        return pd.concat(parts)

    def fit(self, actions, games):
        rng = np.random.default_rng(self.random_state)
        ids = actions.game_id.unique()
        if self.max_train_games and len(ids) > self.max_train_games:
            ids = rng.choice(ids, self.max_train_games, replace=False)
        # hold out whole games (socceraction holds out random actions) for early stopping
        ids = rng.permutation(ids)
        n_val = int(round(len(ids) * self.val_size))
        val_ids, fit_ids = ids[:n_val], ids[n_val:]

        def xy(game_ids):
            a = actions[actions.game_id.isin(game_ids)]
            return self._features(a, games).values, compute_labels(a, games)

        X, y = xy(fit_ids)
        X_val, y_val = xy(val_ids)
        self.models = {}
        for col in ["scores", "concedes"]:
            m = XGBClassifier(**self.tree_params, random_state=self.random_state, n_jobs=2,
                              tree_method="hist")
            m.fit(X, y[col].values, eval_set=[(X_val, y_val[col].values)], verbose=False)
            self.models[col] = m
        return self

    def predict_probabilities(self, actions, games) -> pd.DataFrame:
        X = self._features(actions, games)
        return pd.DataFrame({c: m.predict_proba(X.values)[:, 1] for c, m in self.models.items()},
                            index=actions.index)

    def rate(self, actions, games):
        """VAEP value of every action (offensive + defensive part)."""
        # 1. P(score within 10 actions) and P(concede within 10 actions) for each game state
        p = self.predict_probabilities(actions, games)
        # 2. value = change in those probabilities caused by the action
        comps = vaep_formula(actions, p.scores.values, p.concedes.values)
        self.last_components = comps.join(p)
        return comps["vaep_value"]
