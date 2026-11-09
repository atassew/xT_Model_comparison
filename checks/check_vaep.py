"""Sanity checks of VAEP (xtn/methods/vaep.py), on the 4 folds.

For every fold (train on the other leagues + GER, value the held-out league):
  1. labels: share of game states with a goal / a goal conceded within 10 actions
  2. the two classifiers on the UNSEEN league: AUC and Brier score (vs no skill)
  3. values: mean by action type (goals high, failed moves negative), mean of
     the moves (passes and crosses) by third of the pitch
  4. a team's total over its moves in a match correlates with its npxG in the SAME match

The VAEP values of each fold are saved (results/checks/vaep_values_<league>.parquet).

Run from the xt_npxg folder:  python checks/check_vaep.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn import config, data  # noqa: E402
from xtn.methods import VAEPValuer  # noqa: E402
from xtn.methods.vaep import compute_labels  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checks")


def npxg_per_team_match(shots):
    s = shots[~shots.is_penalty & (shots.period_id < 5)]
    return s.groupby(["game_id", "team_id"]).xg.sum().rename("npxg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    games, actions, shots = data.load(DATA_DIR)
    npxg = npxg_per_team_match(shots)
    rows, by_type = [], {}

    for test_lg in config.TEST_LEAGUES:
        t0 = time.time()
        train = actions[actions.league != test_lg]
        test = actions[actions.league == test_lg]
        assert test_lg not in set(train.league.unique())
        m = VAEPValuer().fit(train, games)
        v = m.rate(test, games)
        comps = m.last_components
        pd.DataFrame({"game_id": test.game_id, "action_id": test.action_id, "team_id": test.team_id,
                      "VAEP": v.astype("float32")}).to_parquet(
            os.path.join(OUT_DIR, f"vaep_values_{test_lg}.parquet"), index=False)

        # 1./2. labels and classifiers on the unseen league
        y = compute_labels(test, games)
        q = {}
        for col in ["scores", "concedes"]:
            p, t = comps[col].values, y[col].values
            q[f"{col}_rate"] = t.mean()
            q[f"{col}_auc"] = roc_auc_score(t, p)
            q[f"{col}_brier"] = brier_score_loss(t, p)
            q[f"{col}_brier_no_skill"] = brier_score_loss(t, np.full(len(t), t.mean()))

        # 3. values
        typ = test.type_name.astype(str)
        is_goal = typ.str.startswith("shot") & (test.result_name == "success")
        label = np.where(is_goal, "goal", typ)
        by_type[test_lg] = v.groupby(label).mean()
        is_move = test.type_name.isin(config.MOVE_TYPES)
        ok = test.result_name == "success"
        third = (test.start_x // (config.FIELD_LENGTH / 3)).clip(0, 2)
        mv = v[is_move].groupby(third[is_move]).mean()

        # 4. same-match correlation with npxG (moves only, as in the study)
        tm = (pd.DataFrame({"game_id": test.game_id, "team_id": test.team_id, "v": v.where(is_move, 0.0)})
              .groupby(["game_id", "team_id"]).v.sum().to_frame()
              .join(npxg, how="left").fillna({"npxg": 0.0}))

        rows.append(dict(
            fold=f"value {test_lg}", train_games=train.game_id.nunique(), **q,
            nan_values=int(v.isna().sum()),
            move_mean_success=v[is_move & ok].mean(), move_mean_failed=v[is_move & ~ok].mean(),
            share_failed_moves_negative=(v[is_move & ~ok] < 0).mean(),
            move_mean_own_third=mv.get(0.0), move_mean_mid_third=mv.get(1.0), move_mean_final_third=mv.get(2.0),
            team_match_mean=tm.v.mean(), corr_same_match_npxg=tm.v.corr(tm.npxg),
            seconds=round(time.time() - t0),
        ))
        print(f"fold value {test_lg} done ({rows[-1]['seconds']} s)", flush=True)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.T.to_string())
    res.to_csv(os.path.join(OUT_DIR, "vaep_checks.csv"), index=False)
    print("\nmean VAEP value by action type:")
    print(pd.DataFrame(by_type).sort_values("ENG", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
