"""Statistical tests of the forecasts (step 6).

All tests use the same 1,994 forecasts. A team's forecasts across a season are
not independent, so every standard error is clustered by team (80 teams), and
the bootstrap resamples whole teams.

Test 1: does xT add anything beyond past npxG?  (Model 2 vs Model 1 = Baseline)
    clark_west      main test, out-of-sample, one-sided (H1: Model 2 forecasts better)
    t_test_c        supporting, in-sample: t-test on c in
                    npxG_next = a_league + b * npxG_avg10 + c * xT_avg10
    bootstrap_drmse effect size: RMSE(Model 2) - RMSE(Model 1) with a 95% CI

Test 2: which method is best?
    diebold_mariano two-sided, squared-error loss, for non-nested forecasts

Test 3: the opponent (Baseline+opp and M3 models)
    clark_west and bootstrap_drmse for the nested pairs, wald_test_terms for the
    in-sample joint test of the added terms, diebold_mariano between the M3s

Clustered standard errors use the usual small-sample factor (as Stata's CR1):
    V = G/(G-1) * (n-1)/(n-k) * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1
with G clusters, n rows and k regressors. For a mean (k = 1) this is
    Var(mean) = G/(G-1) * sum_g (sum_{i in g} (f_i - mean))^2 / n^2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def cluster_ids(preds: pd.DataFrame) -> np.ndarray:
    """One cluster per team (league + team id)."""
    return (preds.league.astype(str) + "_" + preds.team_id.astype(str)).values


def clustered_ols(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS coefficients and their cluster-robust (CR1) standard errors."""
    beta, V, G = clustered_ols_cov(X, y, clusters)
    return beta, np.sqrt(np.diag(V)), G


def clustered_ols_cov(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS coefficients, their CR1 covariance matrix and the number of clusters."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    groups = pd.factorize(clusters)[0]
    G = groups.max() + 1
    scores = np.zeros((G, k))
    np.add.at(scores, groups, X * u[:, None])   # sum_{i in g} x_i u_i, per cluster
    meat = scores.T @ scores
    V = G / (G - 1) * (n - 1) / (n - k) * XtX_inv @ meat @ XtX_inv
    return beta, V, G


def clustered_mean_test(f: np.ndarray, clusters: np.ndarray):
    """Mean of f, its clustered standard error, and t = mean / SE."""
    beta, se, G = clustered_ols(np.ones((len(f), 1)), f, clusters)
    return beta[0], se[0], beta[0] / se[0], G


def clark_west(y, f_small, f_big, clusters) -> dict:
    """Clark and West (2007) test of equal accuracy for NESTED models.

    f_t = (y - f_small)^2 - [(y - f_big)^2 - (f_small - f_big)^2]

    The extra term corrects for the noise the bigger model adds by estimating a
    parameter that is 0 under H0. H0: mean(f) = 0; H1: mean(f) > 0 (the bigger
    model forecasts better). One-sided p-value from the normal distribution.
    """
    f = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    m, se, t, G = clustered_mean_test(f, clusters)
    return dict(cw_mean=m, cw_se=se, cw_t=t, cw_p_one_sided=stats.norm.sf(t), clusters=G)


def t_test_c(panel_rows: pd.DataFrame, method: str, clusters) -> dict:
    """In-sample t-test on c in npxG_next = a_league + b * npxG_avg10 + c * xT_avg10.

    Fitted once on the scored rows. Each league has its own intercept, as each
    league has its own regression in the walk-forward; b and c are shared.
    Two-sided p-value from the t distribution with G - 1 degrees of freedom.
    """
    W = config.WINDOW
    leagues = sorted(panel_rows.league.unique())
    X = np.column_stack([(panel_rows.league == lg).values.astype(float) for lg in leagues]
                        + [panel_rows[f"npxg_avg{W}"].values, panel_rows[f"{method}_avg{W}"].values])
    beta, se, G = clustered_ols(X, panel_rows.npxg.values, clusters)
    t = beta[-1] / se[-1]
    # c per one standard deviation of xT_avg10, to compare methods on different scales
    sd = panel_rows[f"{method}_avg{W}"].std()
    return dict(c=beta[-1], c_se=se[-1], c_t=t, c_p_two_sided=2 * stats.t.sf(abs(t), G - 1),
                b_npxg=beta[-2], c_per_sd=beta[-1] * sd)


def wald_test_terms(panel_rows: pd.DataFrame, base_cols: list[str], test_cols: list[str],
                    clusters) -> dict:
    """In-sample test that the coefficients of ``test_cols`` are all 0.

    npxG_next = a_league + base_cols + test_cols (one OLS on the scored rows,
    one intercept per league, CR1 clustered SEs). Returns each tested
    coefficient with its t and two-sided p (t, G - 1 df), and the joint test
    F = b' V^-1 b / q with an F(q, G - 1) p-value (q = number of tested terms).
    """
    leagues = sorted(panel_rows.league.unique())
    X = np.column_stack([(panel_rows.league == lg).values.astype(float) for lg in leagues]
                        + [panel_rows[c].values for c in base_cols + test_cols])
    beta, V, G = clustered_ols_cov(X, panel_rows.npxg.values, clusters)
    q = len(test_cols)
    b, Vb = beta[-q:], V[-q:, -q:]
    F = float(b @ np.linalg.solve(Vb, b)) / q
    out = dict(F=F, F_p=stats.f.sf(F, q, G - 1))
    for c, bc, se in zip(test_cols, b, np.sqrt(np.diag(Vb))):
        out.update({f"coef_{c}": bc, f"t_{c}": bc / se, f"p_{c}": 2 * stats.t.sf(abs(bc / se), G - 1)})
    return out


def bootstrap_drmse(y, f_small, f_big, clusters, n_boot: int = 2000,
                    seed: int = config.RANDOM_STATE) -> dict:
    """RMSE(big) - RMSE(small) with a team-block bootstrap 95% CI.

    Whole teams are resampled with replacement, n_boot times; the forecasts are
    kept fixed. Negative = the bigger model (Model 2) is more accurate.
    """
    groups = pd.factorize(clusters)[0]
    G = groups.max() + 1
    # per team: sum of squared errors and number of forecasts
    sse_s = np.bincount(groups, (y - f_small) ** 2, G)
    sse_b = np.bincount(groups, (y - f_big) ** 2, G)
    cnt = np.bincount(groups, minlength=G)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    n = cnt[idx].sum(1)
    boot = np.sqrt(sse_b[idx].sum(1) / n) - np.sqrt(sse_s[idx].sum(1) / n)
    d = np.sqrt(sse_b.sum() / cnt.sum()) - np.sqrt(sse_s.sum() / cnt.sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(drmse=d, drmse_ci_low=lo, drmse_ci_high=hi, share_boot_below_0=(boot < 0).mean())


def diebold_mariano(y, f_a, f_b, clusters) -> dict:
    """Diebold and Mariano (1995) test of equal accuracy, squared-error loss.

    d_t = (y - f_a)^2 - (y - f_b)^2. H0: mean(d) = 0. mean(d) < 0 means A is
    more accurate. Clustered SE; two-sided p-value from the normal distribution.
    """
    d = (y - f_a) ** 2 - (y - f_b) ** 2
    m, se, t, G = clustered_mean_test(d, clusters)
    return dict(dm_mean_d=m, dm_se=se, dm_t=t, dm_p_two_sided=2 * stats.norm.sf(abs(t)), clusters=G)
