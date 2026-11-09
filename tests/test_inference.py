"""Checks of the test statistics in xtn/inference.py against hand-computed values.

Run:  python tests/test_inference.py      (or: pytest tests)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xtn import inference  # noqa: E402

RNG = np.random.default_rng(0)


def test_mean_se_one_row_per_cluster_is_the_textbook_se():
    # with every row its own cluster, the clustered SE of a mean is s / sqrt(n)
    f = RNG.normal(size=50)
    _, se, _, _ = inference.clustered_mean_test(f, np.arange(50))
    assert np.isclose(se, f.std(ddof=1) / np.sqrt(50))


def test_mean_se_by_hand_with_two_clusters():
    f = np.array([1.0, 3.0, 2.0, 6.0])        # mean 3
    cl = np.array([0, 0, 1, 1])               # cluster sums of (f - 3): -2 and +2
    _, se, _, G = inference.clustered_mean_test(f, cl)
    assert G == 2
    assert np.isclose(se, np.sqrt(2 / 1 * (4 + 4) / 16))   # G/(G-1) * sum / n^2


def test_ols_coefficients_match_numpy():
    X = np.column_stack([np.ones(40), RNG.normal(size=40)])
    y = 1 + 2 * X[:, 1] + RNG.normal(size=40)
    beta, _, _ = inference.clustered_ols(X, y, np.repeat(np.arange(10), 4))
    assert np.allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0])


def test_clark_west_by_hand():
    y = np.array([1.0, 2.0])
    small, big = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    # f = (y - small)^2 - [(y - big)^2 - (small - big)^2] = [1 - (0 - 1), 4 - (1 - 1)] = [2, 4]
    r = inference.clark_west(y, small, big, np.array([0, 1]))
    assert np.isclose(r["cw_mean"], 3.0)


def test_diebold_mariano_sign():
    # A is perfect, B is off by 1 everywhere: mean d = 0 - 1 = -1 (A better)
    y = RNG.normal(size=30)
    r = inference.diebold_mariano(y, y.copy(), y + 1, np.repeat(np.arange(10), 3))
    assert np.isclose(r["dm_mean_d"], -1.0)


def test_bootstrap_point_estimate_is_the_rmse_difference():
    y = RNG.normal(size=60)
    small, big = y + RNG.normal(size=60), y + 0.5 * RNG.normal(size=60)
    r = inference.bootstrap_drmse(y, small, big, np.repeat(np.arange(20), 3), n_boot=200)
    rmse = lambda f: np.sqrt(np.mean((y - f) ** 2))  # noqa: E731
    assert np.isclose(r["drmse"], rmse(big) - rmse(small))
    assert r["drmse_ci_low"] <= r["drmse"] <= r["drmse_ci_high"]


def test_t_test_c_finds_a_real_effect_and_ignores_noise():
    n = 2000
    df = pd.DataFrame({"league": np.repeat(["A", "B"], n // 2), "team_id": np.repeat(np.arange(80), n // 80),
                       "npxg_avg10": RNG.normal(size=n), "Real_avg10": RNG.normal(size=n),
                       "Noise_avg10": RNG.normal(size=n)})
    df["npxg"] = 1 + 0.5 * df.npxg_avg10 + 0.3 * df.Real_avg10 + RNG.normal(size=n)
    cl = inference.cluster_ids(df)
    assert inference.t_test_c(df, "Real", cl)["c_p_two_sided"] < 0.001
    assert abs(inference.t_test_c(df, "Noise", cl)["c"]) < 0.1


def test_wald_with_one_term_is_the_squared_t():
    # testing a single coefficient: F = t^2, and the same t as t_test_c
    n = 600
    df = pd.DataFrame({"league": np.repeat(["A", "B"], n // 2), "team_id": np.repeat(np.arange(40), n // 40),
                       "npxg_avg10": RNG.normal(size=n), "Real_avg10": RNG.normal(size=n)})
    df["npxg"] = 1 + 0.5 * df.npxg_avg10 + 0.2 * df.Real_avg10 + RNG.normal(size=n)
    cl = inference.cluster_ids(df)
    w = inference.wald_test_terms(df, ["npxg_avg10"], ["Real_avg10"], cl)
    assert np.isclose(w["F"], inference.t_test_c(df, "Real", cl)["c_t"] ** 2)
    assert np.isclose(w["F_p"], w["p_Real_avg10"])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
