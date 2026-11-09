"""Walk-forward forecasts of next-match npxG, and their accuracy.

The models (xT_avg10 = one method's 10-match average: Karun, Sumpter or VAEP)
------------------------------------------------------------------------------
    Baseline        npxG_next = a + b * npxG_avg10              (= Model 1 of Test 1)
    <method>        npxG_next = a + b * xT_avg10                (one per method)
    M2 <method>     npxG_next = a + b * npxG_avg10 + c * xT_avg10   (Model 2, one per method)

With the opponent (oppNpxGc / oppXTc = the opponent's npxG / xT conceded,
averaged over ITS previous 10 matches, i.e. how weak its defence is):

    Baseline+opp    npxG_next = a + b * npxG_avg10 + d * oppNpxGc
    M3 <method>     npxG_next = a + b * npxG_avg10 + d * oppNpxGc + c * xT_avg10 + e * oppXTc

Every model is a linear regression, so all forecasts are on the npxG scale.

Evaluation design
-----------------
* Walk-forward, per league: the season is walked through date by date. Before
  date d, every regression is fitted on that league's rows played BEFORE d and
  then predicts the rows ON d (checked: no training row is on or after d).
* Training rows: every team-match with a 10-match average (the team's 11th match on).
  A model with opponent inputs also needs the opponent's 10-match average; the
  rare training rows without it (2 in the data) are left out of that model only.
  So on FRA's first scored date the opponent models train on 58 rows, not 60
  (accepted exception: FRA's 3 missing games leave two opponents with 9 matches).
* Scored rows: every team's 14th match on. Matches 11-13 are training data only;
  per league they give 3 x 20 = 60 rows, the minimum the regression is fitted on
  (checked for every scored date).
* Same rows: every model is scored on exactly the same forecasts (checked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config

W = config.WINDOW
BASELINE = "Baseline"
BASELINE_OPP = "Baseline+opp"
FIRST_SCORED_MATCH = W + 4   # 14: matches 11-13 are training data only
KEY = ["game_id", "league", "team_id", "opp_id", "game_date", "match_no", "npxg"]


def model_specs(methods=config.METHODS) -> dict[str, list[str]]:
    """Regression models: name -> input columns.

    {"Baseline": ["npxg_avg10"], "Karun": ["Karun_avg10"], ...,
     "M2 Karun": ["npxg_avg10", "Karun_avg10"], ...,
     "Baseline+opp": ["npxg_avg10", "opp_npxg_conc_avg10"],
     "M3 Karun": ["npxg_avg10", "opp_npxg_conc_avg10", "Karun_avg10", "opp_Karun_conc_avg10"], ...}
    """
    specs = {BASELINE: [f"npxg_avg{W}"]}
    specs.update({m: [f"{m}_avg{W}"] for m in methods})
    specs.update({f"M2 {m}": [f"npxg_avg{W}", f"{m}_avg{W}"] for m in methods})
    specs[BASELINE_OPP] = [f"npxg_avg{W}", f"opp_npxg_conc_avg{W}"]
    specs.update({f"M3 {m}": specs[BASELINE_OPP] + [f"{m}_avg{W}", f"opp_{m}_conc_avg{W}"] for m in methods})
    return specs


def walk_forward(panel: pd.DataFrame, specs: dict[str, list[str]],
                 min_train: int = config.MIN_TRAIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward forecasts of next-match npxG.

    Returns
    -------
    preds : one row per scored team-match: the actual npxG and one column per model
    coefs : the fitted coefficients of every model on every scored date
    """
    # rows with the team's own averages; opponent averages may be missing in a few
    # training rows, but must exist in every scored row (checked below)
    own = [f"{c}_avg{W}" for c in config.METHODS + ["npxg"]]
    data = panel.dropna(subset=own + ["npxg"])
    assert (data.match_no > W).all()
    inputs = sorted({c for cols in specs.values() for c in cols})
    assert data.loc[data.match_no >= FIRST_SCORED_MATCH, inputs].notna().all().all(), \
        "a scored row is missing an input"

    out, coefs = [], []
    for league, d in data.groupby("league"):
        d = d.sort_values("game_date")
        for date in d.game_date.unique():
            test = d[(d.game_date == date) & (d.match_no >= FIRST_SCORED_MATCH)]
            if test.empty:
                continue
            train = d[d.game_date < date]      # strictly earlier matches only
            assert train.game_date.max() < date, "look-ahead"
            assert len(train) >= min_train, f"{league} {date}: only {len(train)} training rows"
            res = test[KEY].copy()
            for name, cols in specs.items():
                # only a missing opponent average may drop rows (opponent models only); this can
                # leave fewer than min_train rows (FRA's first scored date: 58), reported in run.py
                tr = train.dropna(subset=cols)
                assert len(tr) == len(train) or any(c.startswith("opp_") for c in cols)
                lr = LinearRegression().fit(tr[cols].values, tr.npxg.values)
                res[name] = lr.predict(test[cols].values)
                coefs.append(dict(league=league, game_date=date, model=name, n_train=len(tr),
                                  a=lr.intercept_, **{f"coef_{c}": b for c, b in zip(cols, lr.coef_)}))
            out.append(res)
    preds = pd.concat(out, ignore_index=True)

    # same rows: every model has a forecast for every scored team-match
    expected = data[data.match_no >= FIRST_SCORED_MATCH]
    assert len(preds) == len(expected), "a scored team-match was not forecast"
    missing = preds[list(specs)].isna().sum()
    assert not missing.any(), f"models without a forecast on some rows: {missing[missing > 0].to_dict()}"
    return preds, pd.DataFrame(coefs)


def metrics(preds: pd.DataFrame, models) -> pd.DataFrame:
    """RMSE and MAE of every model, pooled ("ALL") and per league.

    RMSE  root mean squared error   (lower is better)
    MAE   mean absolute error       (lower is better)
    dRMSE_vs_baseline  RMSE(model) - RMSE(baseline)   (negative = better than the baseline)
    R2    share of the variance of npxG explained: 1 - MSE / var(npxG)   (higher is better)
    """
    rows = []
    for lg, d in [("ALL", preds)] + list(preds.groupby("league")):
        y = d.npxg.values
        base_rmse = np.sqrt(np.mean((y - d[BASELINE].values) ** 2))
        for name in models:
            e = y - d[name].values
            rmse = np.sqrt(np.mean(e ** 2))
            rows.append(dict(league=lg, model=name, n=len(d), RMSE=rmse, MAE=np.mean(np.abs(e)),
                             dRMSE_vs_baseline=rmse - base_rmse, R2=1 - np.mean(e ** 2) / np.var(y)))
    return pd.DataFrame(rows)
