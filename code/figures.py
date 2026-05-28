"""
Generate all figures and LaTeX tables for the paper.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
FIGDIR = OUT / "figures"
TABDIR = OUT / "tables"
FIGDIR.mkdir(exist_ok=True)
TABDIR.mkdir(exist_ok=True)

MAINBLUE = "#003DA5"
IDARED   = "#C8102E"
FOREST   = "#228B22"
CRIMSON  = "#DC143C"
COLORS = {"GJR-GARCH-t": MAINBLUE, "TimesFM-2.5": IDARED,
           "Chronos-Small": FOREST, "Moirai-2.0": CRIMSON}
MARKERS = {"GJR-GARCH-t": "o", "TimesFM-2.5": "s",
           "Chronos-Small": "^", "Moirai-2.0": "D"}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                      "figure.figsize": (8, 6)})


def load():
    df = pd.read_csv(OUT / "data" / "recalib_results.csv")
    roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv",
                        parse_dates=["date"])
    return df, roll


# ── Figure 1: Main scatter ────────────────────────────────────────

def fig_scatter_main(df):
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Plug-in benchmark $\hat{C}/\sqrt{n\alpha}$")
    ax.set_ylabel("Detrended estimation SD")
    ax.set_title("ES recalibration dispersion vs. plug-in rate benchmark")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scatter_main.pdf", bbox_inches="tight",
                transparent=True)
    fig.savefig(FIGDIR / "fig_scatter_main.png", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print("  fig_scatter_main done")


# ── Figure 2: Scatter by forecaster (panels) ─────────────────────

def fig_scatter_by_forecaster(df):
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
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scatter_by_forecaster.pdf", bbox_inches="tight",
                transparent=True)
    fig.savefig(FIGDIR / "fig_scatter_by_forecaster.png", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print("  fig_scatter_by_forecaster done")


# ── Figure 3: Time series for SP500 ──────────────────────────────

def fig_timeseries_sp500(roll, df):
    sub = roll[(roll["asset"] == "SP500") &
               (roll["forecaster"] == "GJR-GARCH-t") &
               (roll["alpha"] == 0.01)].copy()
    sub = sub.sort_values("date")

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
    fig.savefig(FIGDIR / "fig_timeseries_sp500.pdf", bbox_inches="tight",
                transparent=True)
    fig.savefig(FIGDIR / "fig_timeseries_sp500.png", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print("  fig_timeseries_sp500 done")


# ── Figure 4: VaR filter scatter ─────────────────────────────────

def fig_var_filter(df):
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
    ax.set_ylabel("RMSE / Benchmark (detrended)")
    ax.set_title("VaR calibration quality vs. ES benchmark proximity ($\\alpha = 1\\%$)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_var_filter.pdf", bbox_inches="tight",
                transparent=True)
    fig.savefig(FIGDIR / "fig_var_filter.png", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print("  fig_var_filter done")


# ── Table: Tightness summary ─────────────────────────────────────

def tab_tightness(df):
    sub = df[df["alpha"] == 0.01]
    grp = sub.groupby("forecaster")["ratio_detrended"]
    tbl = grp.agg(["median", lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.75), "min", "max"])
    tbl.columns = ["Median", "Q1", "Q3", "Min", "Max"]
    tbl = tbl.round(2)
    tbl.to_csv(TABDIR / "tightness.csv")

    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Forecaster & Median & Q1 & Q3 & Min & Max \\",
             r"\midrule"]
    for fname, row in tbl.iterrows():
        lines.append(f"  {fname} & {row['Median']:.2f} & {row['Q1']:.2f} & "
                     f"{row['Q3']:.2f} & {row['Min']:.2f} & {row['Max']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABDIR / "tightness.tex").write_text("\n".join(lines))
    print("  tightness table done")


# ── Table: Asset breakdown ────────────────────────────────────────

def tab_asset_breakdown(df):
    sub = df[(df["alpha"] == 0.01) & (df["forecaster"] == "GJR-GARCH-t")]
    sub = sub.sort_values("ratio_detrended")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Asset & Detr.\ SD & Bound & Ratio & Kupiec $p$ \\",
             r"\midrule"]
    for _, row in sub.iterrows():
        lines.append(f"  {row['asset']} & {row['detrended_sd']:.4f} & "
                     f"{row['bound']:.4f} & {row['ratio_detrended']:.2f} & "
                     f"{row['kupiec_p']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABDIR / "asset_breakdown.tex").write_text("\n".join(lines))
    sub.to_csv(TABDIR / "asset_breakdown.csv", index=False)
    print("  asset_breakdown table done")


# ── Table: Asset list ─────────────────────────────────────────────

def tab_asset_list():
    assets_info = {
        "SP500": ("S\\&P 500", "Equity"),
        "STOXX": ("Euro Stoxx 50", "Equity"),
        "GDAXI": ("DAX", "Equity"),
        "FCHI": ("CAC 40", "Equity"),
        "FTSE100": ("FTSE 100", "Equity"),
        "NIKKEI": ("Nikkei 225", "Equity"),
        "HSI": ("Hang Seng", "Equity"),
        "BOVESPA": ("Bovespa", "Equity"),
        "NIFTY": ("Nifty 50", "Equity"),
        "ASX200": ("ASX 200", "Equity"),
        "ICLN": ("iShares Clean Energy", "Equity"),
        "TLT": ("US 20Y+ Treasury", "Bond"),
        "IBGL": ("Euro Gov Bond", "Bond"),
        "DJCI": ("DJ Commodity", "Commodity"),
        "GOLD": ("Gold", "Commodity"),
        "WTI": ("WTI Crude", "Commodity"),
        "NATGAS": ("Natural Gas", "Commodity"),
        "CBU0": ("Copper", "Commodity"),
        "BTC": ("Bitcoin", "Crypto"),
        "ETH": ("Ethereum", "Crypto"),
        "EURUSD": ("EUR/USD", "FX"),
        "GBPUSD": ("GBP/USD", "FX"),
        "USDJPY": ("USD/JPY", "FX"),
        "AUDUSD": ("AUD/USD", "FX"),
    }

    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Ticker & Name & Class & Sample \\",
             r"\midrule"]
    for ticker, (name, cls) in assets_info.items():
        try:
            ret = pd.read_csv(
                f"/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data/returns/{ticker}.csv",
                parse_dates=["date"])
            start = ret["date"].min().strftime("%Y-%m")
            end = ret["date"].max().strftime("%Y-%m")
            period = f"{start} -- {end}"
        except:
            period = "---"
        lines.append(f"  {ticker} & {name} & {cls} & {period} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABDIR / "asset_list.tex").write_text("\n".join(lines))
    print("  asset_list table done")


# ── main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures and tables...")
    df, roll = load()

    fig_scatter_main(df)
    fig_scatter_by_forecaster(df)
    fig_timeseries_sp500(roll, df)
    fig_var_filter(df)

    tab_tightness(df)
    tab_asset_breakdown(df)
    tab_asset_list()

    print("\nAll done.")
