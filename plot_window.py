"""Figure: forecast error for next-match npxG against the number of past matches averaged.

Two stacked panels, both on the same axes so they can be read against each other:

    top     xT as the best pass or cross per sequence   (results/window_metrics_long_max.csv)
    bottom  xT summed over the match                    (results/window_metrics_long.csv)

Three model families per panel. Colour is the xT method, line style is the model:

    solid   own xT-for + opponent xT-against
    dotted  all four: own xT-for, opponent xT-against, own npxG-for, opponent npxG-conceded

npxG (steel blue, dotted) is the comparator: own npxG-for + opponent npxG-conceded,
with no xT at all. It is identical in both panels, since it uses no xT.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

CM = 1 / 2.54
COLOURS = {"npxG": "#4E79A7", "Sumpter": "#E15759", "Karun": "#59A14F", "VAEP": "#9B6BB5"}
ORDER = ["npxG", "Sumpter", "Karun", "VAEP"]
# the method keys above are the column names in the metrics CSVs; these are what
# the legend shows instead
LABEL = {"Sumpter": "Soccermatics"}
GREY = "#7A7A7A"
MODEL_STYLE = {"pair": "-", "four": (0, (1.5, 1.5)), "npxG": (0, (6, 2))}
MODEL_LABEL = {
    "pair": "xT-for + opponent xT-against",
    "four": "all four: + npxG-for + opponent npxG-conceded",
    "npxG": "no xT: npxG-for + opponent npxG-conceded",
}
# the comparator is drawn heavier and on top, so it reads against the rest
NPXG_WIDTH = 2.6
# Where the npxG curve stops improving: the team-block bootstrap cannot separate
# RMSE at this window from RMSE at n = 30 (dRMSE -0.0004, 95% CI -0.0095 to +0.0088).
FLAT_FROM = 24

PANELS = [
    ("results/window_metrics_long.csv", "xT = summed over the match"),
]
OUT = "results/window_sensitivity.png"


def legend_handles(xt_methods: list[str]) -> list:
    """Colour entries for the xT methods, then one entry per model family."""
    handles = [plt.Line2D([], [], color=COLOURS[m], linewidth=1.5, label=LABEL.get(m, m))
               for m in xt_methods]
    handles += [
        plt.Line2D([], [], color=GREY, linewidth=1.5, linestyle=MODEL_STYLE[k], label=MODEL_LABEL[k])
        for k in ("pair", "four")
    ]
    handles.append(plt.Line2D([], [], color=COLOURS["npxG"], linewidth=NPXG_WIDTH,
                              linestyle=MODEL_STYLE["npxG"], label=MODEL_LABEL["npxG"]))
    return handles


def panel(ax, metrics_path: str, feature: str) -> int:
    metrics = pd.read_csv(metrics_path)
    pooled = metrics[metrics.league == "ALL"].pivot(index="window", columns="method", values="RMSE")
    methods = [m for m in ORDER if m in pooled.columns]

    ax.axvline(FLAT_FROM, color="#B0B0B0", linewidth=1.0, linestyle=":")

    xt_methods = [m for m in methods if m != "npxG"]
    for method in xt_methods:
        ax.plot(pooled[method].index, pooled[method].values, color=COLOURS[method],
                linewidth=1.5, linestyle=MODEL_STYLE["pair"])
        four = f"{method}+npxG"
        if four in pooled.columns:
            ax.plot(pooled[four].index, pooled[four].values, color=COLOURS[method],
                    linewidth=1.5, linestyle=MODEL_STYLE["four"])
    # drawn last and heaviest, so the comparator is never hidden under an xT curve
    ax.plot(pooled["npxG"].index, pooled["npxG"].values, color=COLOURS["npxG"],
            linewidth=NPXG_WIDTH, linestyle=MODEL_STYLE["npxG"], zorder=5)


    ax.set_title(feature, fontsize=11)
    ax.set_ylabel("RMSE of next-match npxG", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlim(1, pooled.index.max())
    ax.set_xticks([n for n in (1, 5, 10, 15, 20, 25, 30) if n <= pooled.index.max()])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    return int(metrics[metrics.league == "ALL"].n.iloc[0])


def main() -> None:
    missing = [p for p, _ in PANELS if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing metrics: {', '.join(missing)}")

    values = pd.concat([pd.read_csv(p) for p, _ in PANELS]).query("league == 'ALL'").RMSE
    pad = 0.05 * (values.max() - values.min())

    height = 10 if len(PANELS) == 1 else 18
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15 * CM, height * CM),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    n_scored = [panel(ax, path, feature) for ax, (path, feature) in zip(axes, PANELS)]
    axes[0].set_ylim(values.min() - pad, values.max() + pad)
    axes[-1].set_xlabel("Matches averaged, n", fontsize=11)
    axes[0].annotate(f"npxG flat from n ≈ {FLAT_FROM}", xy=(FLAT_FROM - 0.6, 0.97),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=10, color=GREY,
                     ha="right", va="top")
    fig.suptitle(f"Forecasting next-match npxG · {n_scored[0]} team-matches", fontsize=11)
    fig.legend(handles=legend_handles([m for m in ORDER if m != "npxG"]), fontsize=10,
               loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=(0, 0.20 if len(PANELS) == 1 else 0.11, 1, 1))
    plt.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
