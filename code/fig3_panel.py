"""
Figure 3: 2x2 rolling-window ES correction panel.
SP500 (equity), USDJPY (FX), GOLD (commodity), BTC (crypto).
GJR-GARCH-t at alpha=1%, n=250, step=21.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
FIGDIR = OUT / "figures"

MAINBLUE = "#003DA5"
SAVE_KW = dict(bbox_inches="tight", transparent=True)

roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])
res  = pd.read_csv(OUT / "data" / "recalib_results.csv")

PANELS = [
    ("SP500",  "S&P 500 (equity)"),
    ("USDJPY", "USD/JPY (FX)"),
    ("GOLD",   "Gold (commodity)"),
    ("BTC",    "Bitcoin (crypto)"),
]

ANNOTATIONS = {
    "SP500": [("2020-03-15", "COVID\ncrash")],
    "USDJPY": [],
    "GOLD":  [("2022-03-15", "Commodity\ndislocation")],
    "BTC":   [("2022-11-15", "FTX\ncollapse")],
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, (asset, title) in zip(axes.flat, PANELS):
    sub = roll[(roll["asset"] == asset) &
               (roll["forecaster"] == "GJR-GARCH-t") &
               (roll["alpha"] == 0.01)].sort_values("date")

    row = res[(res["asset"] == asset) &
              (res["forecaster"] == "GJR-GARCH-t") &
              (res["alpha"] == 0.01)].iloc[0]

    bound = row["bound"]
    mean_r = row["r_hat_mean"]

    dates = sub["date"].values
    r_hat = sub["r_hat"].values

    ax.plot(dates, r_hat, color=MAINBLUE, lw=0.8, label=r"$\hat{r}_n$")
    ax.axhline(mean_r, color="gray", ls=":", lw=0.8, label=f"Mean")
    ax.fill_between(dates,
                    mean_r - 1.96 * bound,
                    mean_r + 1.96 * bound,
                    color=MAINBLUE, alpha=0.12,
                    label=r"$\pm 1.96\,\hat{C}/\sqrt{n\alpha}$")

    for ann_date, ann_text in ANNOTATIONS.get(asset, []):
        ann_dt = np.datetime64(ann_date)
        if ann_dt >= dates[0] and ann_dt <= dates[-1]:
            idx = np.argmin(np.abs(dates - ann_dt))
            y_val = r_hat[idx]
            ax.annotate(ann_text, xy=(dates[idx], y_val),
                        xytext=(0, 25), textcoords="offset points",
                        fontsize=8, ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="gray", alpha=0.9))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("$\\hat{r}_n$", fontsize=10)

fig.tight_layout()

for ax, (asset, _) in zip(axes.flat, PANELS):
    if asset == "BTC":
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=8)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9, framealpha=0.9,
           loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
fig.savefig(FIGDIR / "fig_timeseries_panel.pdf", **SAVE_KW)
fig.savefig(FIGDIR / "fig_timeseries_panel.png", **SAVE_KW, dpi=150)
plt.close(fig)
print("fig_timeseries_panel saved")
