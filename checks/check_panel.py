"""Checks of the panel (results/panel.csv, built by run.py).

  1. one row per team per match of the forecast leagues, every row has the three method totals
  2. no look-ahead: every 10-match average is recomputed the slow way (the team's
     10 matches with an EARLIER date) and must equal the panel's value
  3. how many rows have a 10-match average (expected: matches 11, 12, ... of every team)
  4. VAEP totals are reproducible: the same as the totals from checks/check_vaep.py
  5. an example: one team's 20th match with its 10 previous matches
  6. how strongly the four 10-match averages are related to each other

Run from the xt_npxg folder:  python checks/check_panel.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn import config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA_DIR = os.path.join(HERE, "..", "..", "data")
W = config.WINDOW
COLS = config.METHODS + ["npxg"]


def main():
    panel = pd.read_csv(os.path.join(RESULTS, "panel.csv"), parse_dates=["game_date"])
    games = pd.read_parquet(os.path.join(DATA_DIR, "games.parquet"))

    # 1. rows
    assert not panel.duplicated(["game_id", "team_id"]).any()
    assert panel[COLS].notna().all().all()
    exp = games[games.league.isin(config.TEST_LEAGUES)].groupby("league").size() * 2
    got = panel.groupby("league").size()
    assert (got == exp.reindex(got.index)).all(), (got, exp)
    print("1. team-matches per league:", got.to_dict())

    # 2. look-ahead: recompute every average from strictly earlier matches
    worst = 0.0
    for _, d in panel.groupby("team_id"):
        d = d.sort_values("game_date")
        dates = d.game_date.values
        for i in range(len(d)):
            earlier = d[dates < dates[i]]
            for c in COLS:
                v = d.iloc[i][f"{c}_avg{W}"]
                if len(earlier) < W:
                    assert np.isnan(v), "average before the team has 10 earlier matches"
                else:
                    worst = max(worst, abs(earlier[c].iloc[-W:].mean() - v))
    assert worst < 1e-6, worst  # rolling() uses running sums: rounding differences ~1e-8
    print(f"2. every {W}-match average equals the mean of the team's {W} EARLIER matches "
          f"(largest difference {worst:.1e}); the current match is never included")

    # 2b. opponent's defence: recompute from the opponent's 10 earlier matches, the slow way
    worst, n = 0.0, 0
    for _, r in panel[panel[f"npxg_avg{W}"].notna()].iterrows():
        o = panel[(panel.team_id == r.opp_id) & (panel.game_date < r.game_date)].sort_values("game_date")
        for c in COLS:
            v = r[f"opp_{c}_conc_avg{W}"]
            if len(o) < W:
                assert np.isnan(v), "opponent average before the opponent has 10 earlier matches"
                continue
            # what the opponent conceded = what ITS opponents created in those games
            created = panel.set_index(["game_id", "team_id"])[c]
            conc = [created[(g, t)] for g, t in zip(o.game_id.iloc[-W:], o.opp_id.iloc[-W:])]
            worst = max(worst, abs(np.mean(conc) - v))
        n += 1
    assert worst < 1e-6, worst
    print(f"2b. every opponent average equals what the opponent's {W} EARLIER opponents created against it "
          f"({n} rows, largest difference {worst:.1e})")

    # 3. rows with an average
    has = panel[f"npxg_avg{W}"].notna()
    exp_rows = panel.groupby("team_id").size().sub(W).clip(lower=0).sum()
    assert has.sum() == exp_rows and (panel.loc[has, "match_no"] > W).all()
    print(f"3. rows with a {W}-match average: {has.sum()} (every team from its {W + 1}th match):",
          panel[has].groupby("league").size().to_dict())

    # 4. VAEP reproducibility
    a = pd.read_parquet(os.path.join(DATA_DIR, "actions.parquet"), columns=["game_id", "action_id", "type_name"])
    diffs = []
    for lg in config.TEST_LEAGUES:
        p = os.path.join(RESULTS, "checks", f"vaep_values_{lg}.parquet")
        if not os.path.exists(p):
            continue
        v = pd.read_parquet(p).merge(a, on=["game_id", "action_id"])
        tm = v.VAEP.where(v.type_name.isin(config.MOVE_TYPES), 0).groupby([v.game_id, v.team_id]).sum()
        m = panel.set_index(["game_id", "team_id"]).VAEP.reindex(tm.index)
        diffs.append(float(np.abs(m - tm).max()))
    if diffs:
        print(f"4. VAEP totals equal those of the separate check run (largest difference {max(diffs):.1e})")

    # 5. example
    team = panel[panel.league == "ENG"].team_id.iloc[0]
    names = pd.concat([games[["home_team_id", "home_team_name"]].set_axis(["id", "name"], axis=1),
                       games[["away_team_id", "away_team_name"]].set_axis(["id", "name"], axis=1)]
                      ).drop_duplicates("id").set_index("id").name
    d = panel[panel.team_id == team].sort_values("match_no")
    row = d[d.match_no == 20].iloc[0]
    prev = d[(d.match_no >= 10) & (d.match_no <= 19)]
    pd.set_option("display.width", 200)
    print(f"\n5. example: {names[team]}, match 20 ({row.game_date.date()} v {names[row.opp_id]})")
    show = prev[["match_no", "game_date"] + COLS].assign(opponent=prev.opp_id.map(names))
    print(show.round({c: 3 for c in COLS}).to_string(index=False))
    print("   mean of the 10 above:", prev[COLS].mean().round(3).to_dict())
    print("   panel's _avg10 in row 20:", {c: round(row[f'{c}_avg{W}'], 3) for c in COLS})
    print(f"   actual npxG in match 20 (the target): {row.npxg:.3f}")

    # 6. relations between the predictors
    print(f"\n6. correlations between the {W}-match averages (rows with an average):")
    print(panel.loc[has, [f"{c}_avg{W}" for c in COLS]].corr().round(2).to_string())


if __name__ == "__main__":
    main()
