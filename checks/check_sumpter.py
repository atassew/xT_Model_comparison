"""Sanity checks of Sumpter's action-based xT (xtn/methods/sumpter_chain_xt.py), on the 4 folds.

For every fold (train on the other leagues + GER, value the held-out league):
  1. chain labels: share of chains / moves that end with a shot, and how many
     chains got their label from the foul rule
  2. model A (P(chain ends with a shot)): AUC and Brier score on the UNSEEN league
  3. model B (E[xG | shot-ending chain]): range of the predictions, share below 0
  4. move values: only moves of the team in possession are valued; the mean value
     rises towards the opponent's goal
  5. a team's total in a match correlates with its npxG in the SAME match

Run from the xt_npxg folder:  python checks/check_sumpter.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn import config, data  # noqa: E402
from xtn.methods import SumpterChainXT  # noqa: E402
from xtn.methods.sumpter_chain_xt import _poly_features, label_chains  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checks")


def npxg_per_team_match(shots):
    s = shots[~shots.is_penalty & (shots.period_id < 5)]
    return s.groupby(["game_id", "team_id"]).xg.sum().rename("npxg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    games, actions, shots = data.load(DATA_DIR)
    npxg = npxg_per_team_match(shots)
    rows, bands = [], {}

    for test_lg in config.TEST_LEAGUES:
        train = actions[actions.league != test_lg]
        test = actions[actions.league == test_lg]
        assert test_lg not in set(train.league.unique())
        m = SumpterChainXT().fit(train, games)

        # 1. chain labels of the unseen league (the labels themselves use no model)
        lab = label_chains(test)
        chains = lab.groupby("chain").agg(shot_end=("shot_end", "first"))
        last_own = lab[lab.team_id == lab.possession_team_id].groupby("chain").tail(1)
        ends_with_shot = last_own.set_index("chain").type_name.eq("shot")
        by_foul_rule = int((chains.shot_end.eq(1) & ~ends_with_shot.reindex(chains.index, fill_value=False)).sum())

        # 2./3. models A and B on the moves of the unseen league
        p = m._moves(test, games)
        X = _poly_features(p).values
        prob = m.clf.predict_proba(X)[:, 1]
        xg_pred = m.reg.predict(X)

        # 4. values
        v = m.rate(test, games)
        is_move = test.type_name.isin(config.MOVE_TYPES)
        own = test.team_id == test.possession_team_id
        assert v[~is_move].isna().all(), "non-moves must not be valued"
        assert v[is_move & own].notna().all(), "every move of the team in possession must be valued"
        band = (test.start_x // (config.FIELD_LENGTH / 3)).clip(0, 2)
        mean_by_third = v.groupby(band).mean()
        bands[test_lg] = v.groupby((test.start_x // (config.FIELD_LENGTH / 16)).clip(0, 15)).mean()

        # 5. same-match correlation with npxG
        tm = (pd.DataFrame({"game_id": test.game_id, "team_id": test.team_id, "v": v.fillna(0)})
              .groupby(["game_id", "team_id"]).v.sum().to_frame()
              .join(npxg, how="left").fillna({"npxg": 0.0}))

        y = p.shot_end.astype(int).values
        rows.append(dict(
            fold=f"value {test_lg}", train_games=train.game_id.nunique(),
            chains=len(chains), share_chains_shot=chains.shot_end.mean(),
            chains_labelled_by_foul_rule=by_foul_rule,
            moves_valued=int(v.notna().sum()),
            moves_not_by_possession_team=int((is_move & ~own).sum()),
            share_moves_shot_end=y.mean(),
            A_auc_unseen=roc_auc_score(y, prob), A_brier_unseen=brier_score_loss(y, prob),
            A_brier_no_skill=brier_score_loss(y, np.full(len(y), y.mean())),
            B_pred_min=xg_pred.min(), B_pred_median=np.median(xg_pred), B_pred_max=xg_pred.max(),
            B_share_below_0=(xg_pred < 0).mean(),
            value_mean=v.mean(), value_max=v.max(),
            mean_own_third=mean_by_third.get(0.0), mean_mid_third=mean_by_third.get(1.0),
            mean_final_third=mean_by_third.get(2.0),
            team_match_mean=tm.v.mean(), corr_same_match_npxg=tm.v.corr(tm.npxg),
        ))
        print(f"fold value {test_lg} done", flush=True)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.T.to_string())
    res.to_csv(os.path.join(OUT_DIR, "sumpter_checks.csv"), index=False)
    print("\nmean move value by x band of the start (16 bands, own goal -> opponent goal), x100:")
    print((pd.DataFrame(bands).T * 100).round(2).to_string())


if __name__ == "__main__":
    main()
