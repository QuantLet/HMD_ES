"""
Regenerate ALL figures used in the paper with consistent styling:
  - Transparent background
  - Legend outside, bottom center
  - Consistent font sizes

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
FIGDIR.mkdir(exist_ok=True)

MAINBLUE = "#003DA5"
IDARED   = "#C8102E"
FOREST   = "#228B22"
CRIMSON  = "#DC143C"
COLORS = {"GJR-GARCH-t": MAINBLUE, "TimesFM-2.5": IDARED,
           "Chronos-Small": FOREST, "Moirai-2.0": CRIMSON}
MARKERS = {"GJR-GARCH-t": "o", "TimesFM-2.5": "s",
           "Chronos-Small": "^", "Moirai-2.0": "D"}

SAVE_KW = dict(bbox_inches="tight", transparent=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
})


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", **SAVE_KW)
    fig.savefig(FIGDIR / f"{name}.png", **SAVE_KW, dpi=150)
    plt.close(fig)
    print(f"  {name} saved")


# ── Load data ────────────────────────────────────────────────────────

df   = pd.read_csv(OUT / "data" / "recalib_results.csv")
roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Main scatter
# ══════════════════════════════════════════════════════════════════════

def fig_scatter_main():
    fig, ax = plt.subplots(figsize=(8, 7))
    lo = df["bound"].min() * 0.5
    hi = max(df["bound"].max(), df["detrended_sd"].max()) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="$y = x$ (benchmark)")

    for fname in COLORS:
        sub = df[df["forecaster"] == fname]
        ax.scatter(sub["bound"], sub["detrended_sd"],
                   c=COLORS[fname], marker=MARKERS[fname],
                   s=35, alpha=0.7, edgecolors="white", linewidth=0.3,
                   label=fname)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Plug-in benchmark $\hat{C}/\sqrt{n\alpha}$")
    ax.set_ylabel("Detrended estimation SD")
    ax.set_title("ES recalibration dispersion vs. plug-in rate benchmark")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=5)
    fig.tight_layout()
    save(fig, "fig_scatter_main")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Scatter by forecaster (2x2 panels)
# ══════════════════════════════════════════════════════════════════════

def fig_scatter_by_forecaster():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    lo = df["bound"].min() * 0.5
    hi = max(df["bound"].max(), df["detrended_sd"].max()) * 2

    for ax, fname in zip(axes.flat, COLORS):
        sub = df[df["forecaster"] == fname]
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
        for alpha_val in [0.01, 0.025, 0.05]:
            s2 = sub[sub["alpha"] == alpha_val]
            label = f"$\\alpha={alpha_val}$"
            ax.scatter(s2["bound"], s2["detrended_sd"],
                       c=COLORS[fname], s=40, alpha=0.7,
                       edgecolors="white", linewidth=0.3, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Benchmark $\hat{C}/\sqrt{n\alpha}$", fontsize=9)
        ax.set_ylabel("Detrended SD", fontsize=9)
        med = sub["ratio_detrended"].median()
        ax.text(0.05, 0.95, f"Median ratio: {med:.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Benchmark tightness by forecaster", fontsize=13, y=1.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    fig.tight_layout()
    save(fig, "fig_scatter_by_forecaster")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Time series panel (SP500, USDJPY, GOLD, BTC)
# ══════════════════════════════════════════════════════════════════════

def fig_timeseries_panel():
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
        row = df[(df["asset"] == asset) &
                 (df["forecaster"] == "GJR-GARCH-t") &
                 (df["alpha"] == 0.01)].iloc[0]
        bound = row["bound"]
        mean_r = row["r_hat_mean"]
        dates = sub["date"].values
        r_hat = sub["r_hat"].values

        ax.plot(dates, r_hat, color=MAINBLUE, lw=0.8, label=r"$\hat{r}_n$")
        ax.axhline(mean_r, color="gray", ls=":", lw=0.8, label="Mean")
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
                            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
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
               loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    save(fig, "fig_timeseries_panel")


# ══════════════════════════════════════════════════════════════════════
# Figure: SP500 rolling time series (single panel, appendix)
# ══════════════════════════════════════════════════════════════════════

def fig_timeseries_sp500():
    sub = roll[(roll["asset"] == "SP500") &
               (roll["forecaster"] == "GJR-GARCH-t") &
               (roll["alpha"] == 0.01)].copy().sort_values("date")
    row = df[(df["asset"] == "SP500") &
             (df["forecaster"] == "GJR-GARCH-t") &
             (df["alpha"] == 0.01)].iloc[0]
    bound = row["bound"]
    mean_r = row["r_hat_mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["date"], sub["r_hat"], color=MAINBLUE, lw=0.8,
            label=r"$\hat{r}_n$ (rolling 250-day)")
    ax.axhline(mean_r, color="gray", ls=":", lw=0.8, label=f"Mean = {mean_r:.4f}")
    ax.fill_between(sub["date"],
                     mean_r - 1.96 * bound,
                     mean_r + 1.96 * bound,
                     color=MAINBLUE, alpha=0.15,
                     label=r"$\pm 1.96 \cdot \hat{C}/\sqrt{n\alpha}$")
    ax.set_xlabel("Date")
    ax.set_ylabel("ES correction $\\hat{r}_n$")
    ax.set_title("S&P 500: Rolling ES recalibration at $\\alpha = 1\\%$")
    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    save(fig, "fig_timeseries_sp500")


# ══════════════════════════════════════════════════════════════════════
# Figure: VaR filter scatter
# ══════════════════════════════════════════════════════════════════════

def fig_var_filter():
    sub = df[df["alpha"] == 0.01].copy()
    fig, ax = plt.subplots(figsize=(8, 6))

    for fname in COLORS:
        s = sub[sub["forecaster"] == fname]
        ax.scatter(s["kupiec_p"], s["ratio_detrended"],
                   c=COLORS[fname], marker=MARKERS[fname],
                   s=45, alpha=0.7, edgecolors="white", linewidth=0.3,
                   label=fname)

    ax.axvline(0.05, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Kupiec VaR backtest $p$-value")
    ax.set_ylabel("Benchmark ratio $R$")
    ax.set_title("VaR calibration quality vs. ES benchmark proximity ($\\alpha = 1\\%$)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4)
    fig.tight_layout()
    save(fig, "fig_var_filter")


# ══════════════════════════════════════════════════════════════════════
# Slope histogram: overlapping vs non-overlapping comparison
# ══════════════════════════════════════════════════════════════════════

def fig_slope_comparison():
    no_csv = OUT / "tables" / "window_scaling_nonoverlap_slopes.csv"
    ov_csv = OUT / "tables" / "window_scaling_full_slopes.csv"

    if not no_csv.exists() or not ov_csv.exists():
        print("  SKIP HMD_SlopeHistogram_Comparison (CSV missing)")
        return

    no_slopes = pd.read_csv(no_csv)
    ov_slopes_all = pd.read_csv(ov_csv)
    ov_slopes = ov_slopes_all[(ov_slopes_all["forecaster"] == "GJR-GARCH-t") &
                              (ov_slopes_all["alpha"] == 0.01)]

    no_sl = no_slopes.dropna(subset=["slope"])["slope"].values
    ov_sl = ov_slopes.dropna(subset=["slope"])["slope"].values

    no_med = np.median(no_sl)
    no_q1, no_q3 = np.percentile(no_sl, [25, 75])
    no_pct = no_slopes.dropna(subset=["slope"])["contains_half"].astype(float).mean() * 100

    ov_med = np.median(ov_sl)
    ov_q1, ov_q3 = np.percentile(ov_sl, [25, 75])

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-1.5, 0.2, 18)
    ax.hist(ov_sl, bins=bins, color="gray", alpha=0.45, edgecolor="white",
            linewidth=0.6, label="Overlapping (step=21)")
    ax.hist(no_sl, bins=bins, color=IDARED, alpha=0.75, edgecolor="white",
            linewidth=0.6, label="Non-overlapping (step=n)")
    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")
    ax.axvline(no_med, color=IDARED, linewidth=2, linestyle="--",
               label=f"Non-overlap median $= {no_med:.2f}$")
    ax.axvline(ov_med, color="gray", linewidth=1.5, linestyle=":",
               label=f"Overlap median $= {ov_med:.2f}$")

    txt = (f"Non-overlapping ({len(no_sl)} assets)\n"
           f"  Median: {no_med:.2f}, IQR: [{no_q1:.2f}, {no_q3:.2f}]\n"
           f"  $-0.5$ in CI: {no_pct:.0f}%\n\n"
           f"Overlapping ({len(ov_sl)} assets)\n"
           f"  Median: {ov_med:.2f}, IQR: [{ov_q1:.2f}, {ov_q3:.2f}]")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            family="monospace")

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of assets", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    save(fig, "HMD_SlopeHistogram_Comparison")


# ══════════════════════════════════════════════════════════════════════
# Panel: non-overlapping window scaling (3 assets, GJR only)
# ══════════════════════════════════════════════════════════════════════

def fig_window_nonoverlap():
    csv_path = OUT / "tables" / "window_scaling_nonoverlap.csv"
    slopes_path = OUT / "tables" / "window_scaling_nonoverlap_slopes.csv"
    if not csv_path.exists() or not slopes_path.exists():
        print("  SKIP HMD_WindowScaling_NonOverlap (CSV missing)")
        return

    ws = pd.read_csv(csv_path)
    slopes = pd.read_csv(slopes_path)

    panel_assets = ["SP500", "BTC", "NATGAS"]
    panel_colors = {"SP500": MAINBLUE, "BTC": IDARED, "NATGAS": CRIMSON}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, asset in zip(axes, panel_assets):
        sub = ws[(ws["asset"] == asset) & (~ws["dropped"])]
        sl_row = slopes[slopes["asset"] == asset]
        if len(sl_row) == 0 or pd.isna(sl_row.iloc[0]["slope"]):
            ax.set_title(f"{asset} (insufficient data)", fontsize=12)
            continue
        sl_row = sl_row.iloc[0]
        c = panel_colors[asset]

        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)

        ax.scatter(x, y, color=c, s=60, zorder=3)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = sl_row["intercept"] + sl_row["slope"] * x_fit
        ax.plot(x_fit, y_fit, color=c, linewidth=1.5, zorder=2,
                label=f"$\\hat{{b}} = {sl_row['slope']:.2f}$")
        y_ref = y.mean() - 0.5 * (x_fit - x.mean())
        ax.plot(x_fit, y_ref, color="gray", linewidth=1.2, linestyle="--",
                label="$b = -0.50$", zorder=1)

        ax.set_xlabel("$\\log(n)$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=11)
        ax.set_title(asset, fontsize=12)
        ci_str = f"[{sl_row['ci_lo']:.2f}, {sl_row['ci_hi']:.2f}]"
        ax.text(0.03, 0.03, f"95% CI: {ci_str}", transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom")

    fig.suptitle("Non-overlapping windows (step = n)", fontsize=12, y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    fig.tight_layout()
    save(fig, "HMD_WindowScaling_NonOverlap")


# ══════════════════════════════════════════════════════════════════════
# Slope histogram: all four forecasters
# ══════════════════════════════════════════════════════════════════════

def fig_slope_all_forecasters():
    slopes_path = OUT / "tables" / "window_scaling_nonoverlap_all_slopes.csv"
    if not slopes_path.exists():
        print("  SKIP HMD_SlopeHistogram_AllForecasters (CSV missing)")
        return

    slopes_df = pd.read_csv(slopes_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.5, 0.2, 20)

    for fname in ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]:
        sub = slopes_df[(slopes_df["forecaster"] == fname) &
                        slopes_df["slope"].notna()]
        if len(sub) == 0:
            continue
        ax.hist(sub["slope"].values, bins=bins, color=COLORS[fname],
                alpha=0.45, edgecolor="white", linewidth=0.6, label=fname)

    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")
    all_valid = slopes_df.dropna(subset=["slope"])
    med = all_valid["slope"].median()
    ax.axvline(med, color="gray", linewidth=1.5, linestyle="--",
               label=f"Pooled median $= {med:.2f}$")

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of (asset, forecaster) pairs", fontsize=12)
    ax.set_title("Non-overlapping window scaling slopes — all forecasters")
    ax.legend(fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    save(fig, "HMD_SlopeHistogram_AllForecasters")


# ══════════════════════════════════════════════════════════════════════
# Panel: window scaling by forecaster (2x2)
# ══════════════════════════════════════════════════════════════════════

def fig_window_all_forecasters():
    csv_path = OUT / "tables" / "window_scaling_nonoverlap_all.csv"
    slopes_path = OUT / "tables" / "window_scaling_nonoverlap_all_slopes.csv"
    if not csv_path.exists() or not slopes_path.exists():
        print("  SKIP HMD_WindowScaling_AllForecasters (CSV missing)")
        return

    ws = pd.read_csv(csv_path)
    slopes_df = pd.read_csv(slopes_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, fname in zip(axes.flat,
                         ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]):
        sub_slopes = slopes_df[(slopes_df["forecaster"] == fname) &
                               slopes_df["slope"].notna()]
        med_slope = sub_slopes["slope"].median() if len(sub_slopes) > 0 else np.nan
        pct_ci = (sub_slopes["contains_half"].astype(float).mean() * 100
                  if len(sub_slopes) > 0 else 0)

        for asset in sorted(ws["asset"].unique()):
            pts = ws[(ws["asset"] == asset) & (ws["forecaster"] == fname) &
                     (~ws["dropped"])]
            if len(pts) < 2:
                continue
            x = np.log(pts["n"].values.astype(float))
            y = np.log(pts["raw_sd"].values)
            ax.plot(x, y, "o-", color=COLORS[fname], alpha=0.3, markersize=3)

        xlim = np.array([np.log(200), np.log(1100)])
        y_ref = -3.0 - 0.5 * (xlim - np.log(500))
        ax.plot(xlim, y_ref, "k--", lw=1.2, alpha=0.5, label="$b = -0.50$")

        ax.set_xlabel("$\\log(n)$", fontsize=10)
        ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=10)
        ax.set_title(f"{fname}  (median $\\hat{{b}} = {med_slope:.2f}$, "
                     f"$-0.5$ in CI: {pct_ci:.0f}%)", fontsize=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1)
    fig.suptitle("Non-overlapping window scaling by forecaster", fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "HMD_WindowScaling_AllForecasters")


# ══════════════════════════════════════════════════════════════════════
# Overlapping window: slope histogram
# ══════════════════════════════════════════════════════════════════════

def fig_slope_overlapping():
    slopes_path = OUT / "tables" / "window_scaling_full_slopes.csv"
    if not slopes_path.exists():
        print("  SKIP HMD_SlopeHistogram (CSV missing)")
        return

    slopes = pd.read_csv(slopes_path)
    slopes = slopes[(slopes["forecaster"] == "GJR-GARCH-t") &
                    (slopes["alpha"] == 0.01)]
    valid = slopes.dropna(subset=["slope"])
    sl = valid["slope"].values
    med = np.median(sl)
    q1, q3 = np.percentile(sl, [25, 75])
    pct = valid["contains_half"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(sl, bins=12, range=(-1.5, 0.0), color=MAINBLUE, alpha=0.7,
            edgecolor="white", linewidth=0.8)
    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")
    ax.axvline(med, color=IDARED, linewidth=2, linestyle="--",
               label=f"Median $= {med:.2f}$")

    txt = (f"Median: {med:.2f}\n"
           f"IQR: [{q1:.2f}, {q3:.2f}]\n"
           f"$-0.5$ in CI: {pct:.0f}%")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of assets", fontsize=12)
    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    save(fig, "HMD_SlopeHistogram")


# ══════════════════════════════════════════════════════════════════════
# Overlapping window: 3-panel scaling figure
# ══════════════════════════════════════════════════════════════════════

def fig_window_overlapping():
    csv_path = OUT / "tables" / "window_scaling_full.csv"
    slopes_path = OUT / "tables" / "window_scaling_full_slopes.csv"
    if not csv_path.exists() or not slopes_path.exists():
        print("  SKIP HMD_WindowScaling (CSV missing)")
        return

    ws_all = pd.read_csv(csv_path)
    ws = ws_all[(ws_all["forecaster"] == "GJR-GARCH-t") &
                (ws_all["alpha"] == 0.01)]
    slopes_all = pd.read_csv(slopes_path)
    slopes = slopes_all[(slopes_all["forecaster"] == "GJR-GARCH-t") &
                        (slopes_all["alpha"] == 0.01)]
    ALPHA = 0.01

    panel_assets = ["SP500", "BTC", "NATGAS"]
    panel_colors = {"SP500": MAINBLUE, "BTC": IDARED, "NATGAS": CRIMSON}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, asset in zip(axes, panel_assets):
        sub = ws[(ws["asset"] == asset) & (~ws["dropped"])]
        sl_row = slopes[slopes["asset"] == asset].iloc[0]
        c = panel_colors[asset]

        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)
        ax.scatter(x, y, color=c, s=50, zorder=3)

        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = sl_row["intercept"] + sl_row["slope"] * x_fit
        ax.plot(x_fit, y_fit, color=c, linewidth=1.5, zorder=2,
                label=f"$\\hat{{b}} = {sl_row['slope']:.2f}$")
        y_ref = y.mean() - 0.5 * (x_fit - x.mean())
        ax.plot(x_fit, y_ref, color="gray", linewidth=1.2, linestyle="--",
                label="$b = -0.50$", zorder=1)

        sigma_tail = sub["sigma_tail"].iloc[0]
        y_bound = np.log(sigma_tail) - 0.5 * np.log(ALPHA) - 0.5 * x_fit
        ax.plot(x_fit, y_bound, color=c, linewidth=1, linestyle=":",
                alpha=0.5, label="$\\hat{C}/\\sqrt{n\\alpha}$", zorder=1)

        ax.set_xlabel("$\\log(n)$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=11)
        ax.set_title(f"{asset}", fontsize=12)
        ci_str = f"[{sl_row['ci_lo']:.2f}, {sl_row['ci_hi']:.2f}]"
        ax.text(0.03, 0.03, f"95% CI: {ci_str}", transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout()
    save(fig, "HMD_WindowScaling")


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Regenerating all figures...")
    fig_scatter_main()
    fig_scatter_by_forecaster()
    fig_timeseries_panel()
    fig_timeseries_sp500()
    fig_var_filter()
    fig_slope_comparison()
    fig_window_nonoverlap()
    fig_slope_all_forecasters()
    fig_window_all_forecasters()
    fig_slope_overlapping()
    fig_window_overlapping()
    print("\nAll figures regenerated.")
