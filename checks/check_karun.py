"""Sanity checks of Karun Singh's xT (xtn/methods/karun_xt.py), on the 4 folds.

For every fold (train on the other leagues + GER, value the held-out league):
  1. the value iteration converges
  2. the grid: values rise towards the opponent's goal and are left-right symmetric
  3. the moves are valued as in Singh's original: successful = xT(end) - xT(start),
     failed = 0, everything else not valued
  4. a team's total xT in a match correlates with its npxG in the SAME match

Run from the xt_npxg folder:  python checks/check_karun.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn import config, data  # noqa: E402
from xtn.methods import KarunXT  # noqa: E402
from xtn.methods.karun_xt import MOVE_TYPES as KARUN_MOVES  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "checks")


def npxg_per_team_match(shots):
    s = shots[~shots.is_penalty & (shots.period_id < 5)]
    return s.groupby(["game_id", "team_id"]).xg.sum().rename("npxg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, actions, shots = data.load(DATA_DIR)
    npxg = npxg_per_team_match(shots)
    grids, rows = {}, []

    for test_lg in config.TEST_LEAGUES:
        train = actions[actions.league != test_lg]
        test = actions[actions.league == test_lg]
        assert test_lg not in set(train.league.unique())
        xt = KarunXT().fit(train)
        g = xt.grid
        grids[test_lg] = g
        np.save(os.path.join(OUT_DIR, f"karun_grid_without_{test_lg}.npy"), g)

        # 2. shape of the grid
        col_mean = g.mean(axis=0)                       # mean xT per x band (own goal -> opponent goal)
        centre = g[g.shape[0] // 2 - 1: g.shape[0] // 2 + 1].mean(axis=0)  # the two central y bands
        asym = np.abs(g - g[::-1]).max()                # left-right symmetry
        r, c = np.unravel_index(g.argmax(), g.shape)

        # 3. how moves are valued
        v = xt.rate(test)
        is_move = test.type_name.isin(KARUN_MOVES)
        ok = is_move & (test.result_name == "success")
        assert v[~is_move].isna().all(), "non-moves must not be valued"
        assert v[is_move & ~ok].isna().all(), "failed moves must not be valued (count as 0)"
        assert v[ok].notna().all(), "every successful move must be valued"

        # 4. same-match correlation with npxG (only passes and crosses count, as in the study)
        summed = v.fillna(0).where(test.type_name.isin(config.MOVE_TYPES), 0.0)
        tm = (pd.DataFrame({"game_id": test.game_id, "team_id": test.team_id, "xT": summed})
              .groupby(["game_id", "team_id"]).xT.sum().to_frame()
              .join(npxg, how="left").fillna({"npxg": 0.0}))

        rows.append(dict(
            fold=f"value {test_lg}", train_games=train.game_id.nunique(), n_iter=xt.n_iter,
            max_xT=g.max(), max_at=f"x band {c + 1}/16, y band {r + 1}/12",
            own_third=col_mean[:5].mean(), mid_third=col_mean[5:11].mean(), final_third=col_mean[11:].mean(),
            centre_rises_monotone=bool(np.all(np.diff(centre) >= -1e-4)),
            max_left_right_diff=asym,
            moves_valued=int(ok.sum()), mean_xT_per_success=v[ok].mean(),
            share_positive=(v[ok] > 0).mean(),
            team_match_xT_mean=tm.xT.mean(), corr_same_match_npxg=tm.xT.corr(tm.npxg),
        ))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.T.to_string())
    res.to_csv(os.path.join(OUT_DIR, "karun_checks.csv"), index=False)

    # stability: how much do the 4 grids differ?
    stack = np.stack(list(grids.values()))
    print(f"\nlargest difference between the 4 fold grids in any zone: {np.ptp(stack, axis=0).max():.4f}")
    print("\nxT grid of fold 'value ENG' (rows = y bands, columns = x bands; attack -> right), x100:")
    print(pd.DataFrame(grids["ENG"] * 100).round(1).to_string())


if __name__ == "__main__":
    main()
