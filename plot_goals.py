"""Figure: forecast error for next-match GOALS against the number of past matches averaged.

The companion to plot_window.py. Everything about the forecast is identical --
same action values, same rolling averages, same walk-forward fit, same scored
set -- only the target changes, from non-penalty xG to non-penalty goals.

Colour is the xT method, line style is the model:

    solid   own xT-for + opponent xT-against
    dotted  all four: own xT-for, opponent xT-against, own npxG-for, opponent npxG-conceded

Two comparators carry no xT at all, and are drawn heavier so they read against
the rest: past npxG (steel blue, long dashes) and past goals (amber, dash-dot).
The grey line is the yardstick -- the error from always predicting the average.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

CM = 1 / 2.54
COLOURS = {"npxG": "#4E79A7", "Sumpter": "#E15759", "Karun": "#59A14F",
           "VAEP": "#9B6BB5", "goals": "#F28E2B"}
ORDER = ["npxG", "goals", "Sumpter", "Karun", "VAEP"]
# the method keys above are the column names in the metrics CSVs; these are what
# the legend shows instead
LABEL = {"Sumpter": "Soccermatics"}
GREY = "#7A7A7A"
MODEL_STYLE = {"pair": "-", "four": (0, (1.5, 1.5)),
               "npxG": (0, (6, 2)), "goals": (0, (5, 1.6, 1, 1.6))}
MODEL_LABEL = {
    "pair": "xT (for + opponent against)",
    "four": "xT + npxG (all four terms)",
    "npxG": "npxG only (no xT)",
    "goals": "goals only (no xT)",
}
# the two comparators are drawn heavier and on top
WIDE = 2.6
# RMSE of always predicting the mean of the scored set: the no-skill ceiling
YARDSTICK = 1.2645

SRC = "results/window_metrics_goals.csv"
OUT = "results/window_sensitivity_goals.png"


def legend_handles(xt_methods: list[str]) -> list:
    handles = [plt.Line2D([], [], color=COLOURS[m], linewidth=1.5, label=LABEL.get(m, m))
               for m in xt_methods]
    handles += [
        plt.Line2D([], [], color=GREY, linewidth=1.5, linestyle=MODEL_STYLE[k], label=MODEL_LABEL[k])
        for k in ("pair", "four")
    ]
    handles += [
        plt.Line2D([], [], color=COLOURS[k], linewidth=WIDE, linestyle=MODEL_STYLE[k],
                   label=MODEL_LABEL[k])
        for k in ("npxG", "goals")
    ]
    return handles


def main() -> None:
    if not os.path.exists(SRC):
        raise SystemExit(f"missing metrics: {SRC}")

    metrics = pd.read_csv(SRC)
    pooled = metrics[metrics.league == "ALL"].pivot(index="window", columns="method", values="RMSE")
    methods = [m for m in ORDER if m in pooled.columns]
    xt_methods = [m for m in methods if m not in ("npxG", "goals")]

    fig, ax = plt.subplots(figsize=(15 * CM, 10 * CM))

    ax.axhline(YARDSTICK, color="#B0B0B0", linewidth=1.0, linestyle=":")
    ax.annotate("no model: always predict the average", xy=(pooled.index.max(), YARDSTICK),
                xytext=(0, -14), textcoords="offset points", fontsize=10, color=GREY,
                ha="right", va="bottom")

    for method in xt_methods:
        ax.plot(pooled[method].index, pooled[method].values, color=COLOURS[method],
                linewidth=1.5, linestyle=MODEL_STYLE["pair"])
        four = f"{method}+npxG"
        if four in pooled.columns:
            ax.plot(pooled[four].index, pooled[four].values, color=COLOURS[method],
                    linewidth=1.5, linestyle=MODEL_STYLE["four"])
    # drawn last and heaviest, so neither comparator hides under an xT curve
    for key in ("npxG", "goals"):
        ax.plot(pooled[key].index, pooled[key].values, color=COLOURS[key],
                linewidth=WIDE, linestyle=MODEL_STYLE[key], zorder=5)

    n_scored = int(metrics[metrics.league == "ALL"].n.iloc[0])
    ax.set_xlabel("Matches averaged, n", fontsize=11)
    ax.set_ylabel("RMSE of next-match goals", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlim(1, pooled.index.max())
    ax.set_xticks([n for n in (1, 5, 10, 15, 20, 25, 30) if n <= pooled.index.max()])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    fig.suptitle(f"Forecasting next-match goals · {n_scored} team-matches", fontsize=11)
    fig.legend(handles=legend_handles(xt_methods), fontsize=10, loc="lower center",
               frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=(0, 0.22, 1, 1))
    plt.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
