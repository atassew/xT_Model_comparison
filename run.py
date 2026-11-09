"""Can a team's average xT over its last 10 matches predict its non-penalty xG
(npxG) in the next match?

Steps
-----
1. Load the tables built from the raw StatsBomb JSON (games, actions, shots).
2. Leave-one-league-out: for each forecast league, fit Karun, Sumpter and VAEP
   on the OTHER leagues (GER included), then value every action of that league.
3. Sum each team's pass and cross values per match (``features.team_match_values``).
4. Panel: npxG and the three totals per team-match, plus each one's average over
   the team's previous 10 matches (``features.build_panel``).
5. Walk-forward forecasts per league (baseline, Karun, Sumpter, VAEP and the three
   Model 2s) on the same rows, scored with RMSE and MAE (``forecast``).
6. Tests (``inference``), all clustered by team:
   Test 1, per method, Model 2 vs Model 1 (baseline): Clark-West (main),
   in-sample t-test on c, team-block bootstrap of the RMSE difference.
   Test 2, Diebold-Mariano for each pair of methods, alone and as Model 2.
   Test 3, the opponent: Baseline+opp vs Baseline, M3 vs Baseline+opp (main),
   M3 vs M2 (Clark-West, bootstrap, in-sample joint test of the added terms),
   and Diebold-Mariano between the three M3s.
   Test 4, each method alone vs the baseline (Diebold-Mariano, bootstrap).

Usage (from the xt_npxg folder)
-----
    python run.py              # uses results/team_match_values.csv if it exists
    python run.py --refit      # refit the three methods (about 25 min, VAEP is slow)
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

import itertools

from xtn import config, data, features, forecast, inference
from xtn.methods import KarunXT, SumpterChainXT, VAEPValuer

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"), help="folder with the data tables")
    p.add_argument("--out", default=os.path.join(HERE, "results"), help="folder for the results")
    p.add_argument("--refit", action="store_true", help="refit the methods even if values are cached")
    p.add_argument("--n-boot", type=int, default=2000, help="team-block bootstrap resamples")
    return p.parse_args()


def value_one_fold(test_lg, games, actions):
    """Fit the three methods WITHOUT the test league, then value the test league.

    Returns one row per team per test-league match with columns Karun, Sumpter, VAEP.
    """
    train = actions[actions.league != test_lg]
    test = actions[actions.league == test_lg]
    # unseen league: the methods never see the league they value
    assert test_lg not in set(train.league.unique()) and set(test.league.unique()) == {test_lg}

    tables = []
    for name, method in [("Karun", KarunXT()), ("Sumpter", SumpterChainXT()), ("VAEP", VAEPValuer())]:
        t0 = time.time()
        values = method.fit(train, games).rate(test, games)
        tables.append(features.team_match_values(test, values, games, name).set_index(["game_id", "team_id"]))
        print(f"  {name:<8} {time.time() - t0:6.1f}s", flush=True)
    return pd.concat(tables, axis=1).reset_index()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    games, actions, shots = data.load(args.data_dir)

    # Steps 2-3: per-team-match method totals (cached, VAEP takes ~25 min)
    values_path = os.path.join(args.out, "team_match_values.csv")
    if os.path.exists(values_path) and not args.refit:
        print(f"using cached {values_path}")
        values_tm = pd.read_csv(values_path)
    else:
        parts = []
        for lg in config.TEST_LEAGUES:
            print(f"[fold] value {lg}: methods fitted on the other leagues", flush=True)
            parts.append(value_one_fold(lg, games, actions))
        values_tm = pd.concat(parts, ignore_index=True)
        values_tm.to_csv(values_path, index=False)

    # Step 4: panel (forecast leagues only; GER is training data only)
    xg_tm = features.team_match_xg(shots, games)
    xg_tm = xg_tm[xg_tm.league.isin(config.TEST_LEAGUES)]
    panel = features.build_panel(xg_tm, values_tm)
    panel.to_csv(os.path.join(args.out, "panel.csv"), index=False)
    print(f"panel: {len(panel)} team-matches, "
          f"{panel[f'npxg_avg{config.WINDOW}'].notna().sum()} with a {config.WINDOW}-match average")

    # Step 5: walk-forward forecasts and their accuracy
    specs = forecast.model_specs()
    preds, coefs = forecast.walk_forward(panel, specs)
    preds.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    coefs.to_csv(os.path.join(args.out, "coefficients.csv"), index=False)
    few = coefs[coefs.n_train < config.MIN_TRAIN]
    if len(few):
        print(f"fitted on fewer than {config.MIN_TRAIN} rows (missing opponent averages):")
        print(few.groupby(["league", "game_date", "n_train"]).model.apply(", ".join).to_string())
    met = forecast.metrics(preds, list(specs))
    met.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    print(f"\nforecasts: {len(preds)} (scored from each team's {forecast.FIRST_SCORED_MATCH}th match):",
          preds.groupby("league").size().to_dict())
    pd.set_option("display.width", 200)
    print(met[met.league == "ALL"].drop(columns="league").round(4).to_string(index=False))

    # Step 6: tests (forecasts fixed; the in-sample t-test uses the same rows)
    rows = preds.merge(panel[["game_id", "team_id"] + [c for c in panel if c.endswith(f"_avg{config.WINDOW}")]],
                       on=["game_id", "team_id"], validate="one_to_one")
    y, cl = rows.npxg.values, inference.cluster_ids(rows)
    base = rows[forecast.BASELINE].values

    test1 = []
    for m in config.METHODS:
        m2 = rows[f"M2 {m}"].values
        test1.append(dict(method=m, **inference.clark_west(y, base, m2, cl),
                          **inference.t_test_c(rows, m, cl),
                          **inference.bootstrap_drmse(y, base, m2, cl, n_boot=args.n_boot)))
    test1 = pd.DataFrame(test1)
    test1.to_csv(os.path.join(args.out, "test1_model2_vs_model1.csv"), index=False)

    test2 = []
    for prefix, label in [("", "alone"), ("M2 ", "Model 2")]:
        for a, b in itertools.combinations(config.METHODS, 2):
            r = inference.diebold_mariano(y, rows[prefix + a].values, rows[prefix + b].values, cl)
            test2.append(dict(models=label, comparison=f"{a} vs {b}", **r,
                              better=a if r["dm_mean_d"] < 0 else b))
    test2 = pd.DataFrame(test2)
    test2.to_csv(os.path.join(args.out, "test2_diebold_mariano.csv"), index=False)

    # Test 3: the opponent. Each pair is nested: (name, smaller model, bigger model, added terms).
    W = config.WINDOW
    pairs = [("Baseline+opp vs Baseline", forecast.BASELINE, forecast.BASELINE_OPP)]
    pairs += [(f"M3 {m} vs Baseline+opp", forecast.BASELINE_OPP, f"M3 {m}") for m in config.METHODS]
    pairs += [(f"M3 {m} vs M2 {m}", f"M2 {m}", f"M3 {m}") for m in config.METHODS]
    test3 = []
    for label, small, big in pairs:
        added = [c for c in specs[big] if c not in specs[small]]
        r = dict(comparison=label, added=" + ".join(added),
                 **inference.clark_west(y, rows[small].values, rows[big].values, cl),
                 **inference.bootstrap_drmse(y, rows[small].values, rows[big].values, cl, n_boot=args.n_boot))
        w = inference.wald_test_terms(rows, specs[small], added, cl)
        r.update(F=w["F"], F_p=w["F_p"])
        r.update({k.replace(f"_avg{W}", ""): v for k, v in w.items() if k.startswith(("coef_", "t_", "p_"))})
        test3.append(r)
    for a, b in itertools.combinations(config.METHODS, 2):
        r = inference.diebold_mariano(y, rows[f"M3 {a}"].values, rows[f"M3 {b}"].values, cl)
        test3.append(dict(comparison=f"M3 {a} vs M3 {b}", **r, better=a if r["dm_mean_d"] < 0 else b))
    test3 = pd.DataFrame(test3)
    test3.to_csv(os.path.join(args.out, "test3_opponent.csv"), index=False)

    # Test 4: each method alone vs the baseline (not nested: Diebold-Mariano + bootstrap)
    test4 = []
    for m in config.METHODS:
        r = inference.diebold_mariano(y, rows[m].values, base, cl)
        test4.append(dict(method=m, **r, better=m if r["dm_mean_d"] < 0 else forecast.BASELINE,
                          **inference.bootstrap_drmse(y, base, rows[m].values, cl, n_boot=args.n_boot)))
    test4 = pd.DataFrame(test4)
    test4.to_csv(os.path.join(args.out, "test4_alone_vs_baseline.csv"), index=False)

    print("\nTest 1: Model 2 (past npxG + method) vs Model 1 (past npxG)")
    print(test1[["method", "cw_t", "cw_p_one_sided", "c", "c_t", "c_p_two_sided",
                 "drmse", "drmse_ci_low", "drmse_ci_high"]].round(4).to_string(index=False))
    print("\nTest 2: Diebold-Mariano (mean d < 0: the first method is more accurate)")
    print(test2[["models", "comparison", "dm_mean_d", "dm_t", "dm_p_two_sided", "better"]]
          .round(4).to_string(index=False))
    nested = test3[test3.added.notna()]
    print("\nTest 3: the opponent (nested pairs: Clark-West, bootstrap, in-sample joint F of the added terms)")
    print(nested[["comparison", "cw_t", "cw_p_one_sided", "drmse", "drmse_ci_low", "drmse_ci_high", "F", "F_p"]]
          .round(4).to_string(index=False))
    print(test3[test3.added.isna()][["comparison", "dm_mean_d", "dm_t", "dm_p_two_sided", "better"]]
          .round(4).to_string(index=False))
    print("\nTest 4: each method alone vs the baseline (Diebold-Mariano; mean d < 0: the method is better)")
    print(test4[["method", "dm_mean_d", "dm_t", "dm_p_two_sided", "drmse", "drmse_ci_low", "drmse_ci_high"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
