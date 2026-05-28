"""
Task 2: Christoffersen conditional-coverage DQ diagnostic.
Computes CC test statistic for each (asset, forecaster) at alpha=1%,
creates scatter vs benchmark ratio R, Theil-Sen fit, Spearman rho.
Also builds R>1 case audit table.

Deterministic seed: 42
Author: Daniel Traian Pele
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from scipy.optimize import minimize
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data")
OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
from scipy.stats import norm, t as student_t

RETURNS_DIR = BASE / "returns"
GJR_DIR = BASE / "benchmarks"
TIMESFM_DIR = BASE / "timesfm25"
CHRONOS_DIR = BASE / "chronos_small"
MOIRAI_DIR = BASE / "moirai2"

ASSETS = [
    "SP500", "STOXX", "GDAXI", "FCHI", "FTSE100", "NIKKEI", "HSI",
    "BOVESPA", "NIFTY", "ASX200", "ICLN",
    "TLT", "IBGL",
    "DJCI", "GOLD", "WTI", "NATGAS", "CBU0",
    "BTC", "ETH",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
]

FORECASTERS = {
    "GJR-GARCH-t": "gjr",
    "TimesFM-2.5": "timesfm",
    "Chronos-Small": "chronos",
    "Moirai-2.0": "moirai",
}

ALPHA = 0.01

COLORS = {"GJR-GARCH-t": "#003DA5", "TimesFM-2.5": "#C8102E",
           "Chronos-Small": "#228B22", "Moirai-2.0": "#DC143C"}
MARKERS = {"GJR-GARCH-t": "o", "TimesFM-2.5": "s",
           "Chronos-Small": "^", "Moirai-2.0": "D"}


def _student_t_es(alpha, df_val, mean, std):
    df_val = np.clip(df_val, 2.1, 200)
    q = student_t.ppf(alpha, df_val)
    pdf_q = student_t.pdf(q, df_val)
    es_std = -(pdf_q / alpha) * ((df_val + q**2) / (df_val - 1))
    return mean + std * es_std


def load_returns(asset):
    df = pd.read_csv(RETURNS_DIR / f"{asset}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()["log_return"]


def load_forecasts(asset, model_key, alpha):
    var_col = f"VaR_{alpha}"
    if model_key == "gjr":
        df = pd.read_parquet(GJR_DIR / f"{asset}_gjr_garch.parquet")
        return df[var_col].copy()
    elif model_key == "timesfm":
        df = pd.read_parquet(TIMESFM_DIR / f"{asset}.parquet")
        return -df[var_col].abs()
    elif model_key == "chronos":
        df = pd.read_parquet(CHRONOS_DIR / f"{asset}.parquet")
        v = df[var_col].copy()
        return -v.abs() if v.median() > 0 else v
    elif model_key == "moirai":
        df = pd.read_parquet(MOIRAI_DIR / f"{asset}.parquet")
        return -df[var_col].abs()
    raise ValueError(model_key)


def christoffersen_cc(hits):
    """Christoffersen (1998) conditional-coverage test.
    Returns (CC_stat, CC_pval, UC_stat, IND_stat)."""
    n = len(hits)
    h = hits.astype(int)
    n1 = h.sum()
    n0 = n - n1

    # Kupiec UC
    pi_hat = n1 / n
    if pi_hat == 0 or pi_hat == 1:
        return np.nan, np.nan, np.nan, np.nan

    uc_stat = -2 * (n0 * np.log(1 - ALPHA) + n1 * np.log(ALPHA)
                     - n0 * np.log(1 - pi_hat) - n1 * np.log(pi_hat))

    # Markov transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if h[i-1] == 0 and h[i] == 0: n00 += 1
        elif h[i-1] == 0 and h[i] == 1: n01 += 1
        elif h[i-1] == 1 and h[i] == 0: n10 += 1
        else: n11 += 1

    # Independence test
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan, uc_stat, np.nan

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi01 == 0 or pi01 == 1 or pi11 == 0 or pi11 == 1 or pi_all == 0 or pi_all == 1:
        return np.nan, np.nan, uc_stat, np.nan

    log_l_ind = (n00 * np.log(1 - pi_all) + (n01 + n11) * np.log(pi_all)
                 + n10 * np.log(1 - pi_all))
    log_l_dep = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
                 + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    ind_stat = -2 * (log_l_ind - log_l_dep)

    cc_stat = uc_stat + max(ind_stat, 0)
    cc_pval = 1 - chi2.cdf(cc_stat, df=2)

    return cc_stat, cc_pval, uc_stat, max(ind_stat, 0)


# Load recalib results for R values
recalib = pd.read_csv(OUT / "data" / "recalib_results.csv")

# Compute CC test for each (asset, forecaster) at alpha=1%
results = []
for asset in ASSETS:
    try:
        ret = load_returns(asset)
    except Exception as e:
        print(f"  SKIP {asset}: {e}")
        continue

    for fname, fkey in FORECASTERS.items():
        try:
            var_hat = load_forecasts(asset, fkey, ALPHA)
        except Exception as e:
            print(f"  SKIP {asset}/{fname}: {e}")
            continue

        idx = ret.index.intersection(var_hat.index).sort_values()
        X = ret.loc[idx].values
        V = var_hat.loc[idx].values
        hits = X <= V

        cc_stat, cc_pval, uc_stat, ind_stat = christoffersen_cc(hits)

        # Get R from recalib results
        rc = recalib[(recalib["asset"] == asset) &
                     (recalib["forecaster"] == fname) &
                     (recalib["alpha"] == ALPHA)]
        R = rc["ratio_detrended"].values[0] if len(rc) > 0 else np.nan
        kupiec_p = rc["kupiec_p"].values[0] if len(rc) > 0 else np.nan
        sigma_tail = rc["sigma_tail"].values[0] if len(rc) > 0 else np.nan

        results.append({
            "asset": asset, "forecaster": fname,
            "cc_stat": cc_stat, "cc_pval": cc_pval,
            "uc_stat": uc_stat, "ind_stat": ind_stat,
            "kupiec_p": kupiec_p,
            "ratio_R": R, "sigma_tail": sigma_tail,
            "n_obs": len(X), "n_hits": int(hits.sum()),
            "hit_rate": hits.mean(),
        })

df_cc = pd.DataFrame(results)
df_cc.to_csv(OUT / "tables" / "christoffersen_diagnostic.csv", index=False)

print(f"\nComputed CC test for {len(df_cc)} (asset, forecaster) pairs")
print(f"CC stat range: [{df_cc['cc_stat'].min():.2f}, {df_cc['cc_stat'].max():.2f}]")

# --- Scatter plot: log(CC stat) vs R ---
valid = df_cc.dropna(subset=["cc_stat", "ratio_R"]).copy()
valid["log_cc"] = np.log(valid["cc_stat"].clip(lower=0.01))

fig, ax = plt.subplots(figsize=(8, 6))

for fname in COLORS:
    sub = valid[valid["forecaster"] == fname]
    ax.scatter(sub["log_cc"], sub["ratio_R"],
               c=COLORS[fname], marker=MARKERS[fname],
               s=50, alpha=0.7, edgecolors="white", linewidth=0.3,
               label=fname)

# Theil-Sen line
from scipy.stats import theilslopes
slope, intercept, lo_slope, hi_slope = theilslopes(valid["ratio_R"], valid["log_cc"])
x_range = np.linspace(valid["log_cc"].min(), valid["log_cc"].max(), 100)
ax.plot(x_range, intercept + slope * x_range, "k-", lw=1.5, alpha=0.7,
        label=f"Theil-Sen: $R = {intercept:.2f} + {slope:.2f}\\,\\log(\\mathrm{{CC}})$")

# Spearman rank correlation
rho, p_spearman = spearmanr(valid["cc_stat"], valid["ratio_R"])
ax.text(0.02, 0.98, f"Spearman $\\rho = {rho:.3f}$, $p = {p_spearman:.3f}$",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("$\\log(\\mathrm{CC\\ statistic})$", fontsize=11)
ax.set_ylabel("Benchmark ratio $R$", fontsize=11)
ax.set_title("Christoffersen CC statistic vs.\\ benchmark ratio ($\\alpha = 1\\%$)")
ax.legend(fontsize=8.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig_cc_diagnostic.pdf", bbox_inches="tight",
            transparent=True)
fig.savefig(OUT / "figures" / "fig_cc_diagnostic.png", bbox_inches="tight",
            transparent=True)
plt.close(fig)

print(f"\nSpearman rho = {rho:.3f}, p = {p_spearman:.4f}")
print(f"Theil-Sen slope = {slope:.3f}, intercept = {intercept:.3f}")

# --- R>1 case audit ---
excess = df_cc[df_cc["ratio_R"] > 1.0].copy()
excess = excess.sort_values("ratio_R", ascending=False)

# Determine qualitative cause
def classify_cause(row):
    if row["kupiec_p"] < 0.001:
        return "Severe VaR miscalibration"
    elif row["kupiec_p"] < 0.05:
        return "VaR miscalibration"
    elif row["ind_stat"] > 5.99:
        return "Serial dependence in violations"
    elif row["hit_rate"] > 3 * ALPHA:
        return "Fat-tail underestimation"
    else:
        return "Regime shift / non-stationarity"

excess["cause"] = excess.apply(classify_cause, axis=1)

print(f"\n{'='*75}")
print(f"R > 1 case audit: {len(excess)} cells")
print(f"{'='*75}")
for _, row in excess.iterrows():
    print(f"  {row['asset']:>8s}/{row['forecaster']:<16s}: R={row['ratio_R']:.2f}, "
          f"Kupiec_p={row['kupiec_p']:.4f}, CC={row['cc_stat']:.1f}, "
          f"sigma_tail={row['sigma_tail']:.4f}, cause={row['cause']}")

# Generate LaTeX audit table
tex_lines = [
    r"\begin{tabular}{llrrrrl}",
    r"\toprule",
    r"Asset & Forecaster & $R$ & Kupiec $p$ & CC stat & $\hat{\sigma}_{\mathrm{tail}}$ & Cause \\",
    r"\midrule",
]
for _, row in excess.iterrows():
    kupiec_fmt = f"{row['kupiec_p']:.3f}" if row['kupiec_p'] >= 0.001 else "$<$0.001"
    tex_lines.append(
        f"  {row['asset']} & {row['forecaster']} & "
        f"{row['ratio_R']:.2f} & {kupiec_fmt} & "
        f"{row['cc_stat']:.1f} & {row['sigma_tail']:.4f} & "
        f"{row['cause']} \\\\"
    )
tex_lines += [r"\bottomrule", r"\end{tabular}"]
tex = "\n".join(tex_lines)
(OUT / "tables" / "excess_dispersion_audit.tex").write_text(tex)
print(f"\nAudit table saved to tables/excess_dispersion_audit.tex")

# Generate LaTeX for VaR diagnostic regression (updated)
tex_diag = r"""\begin{tabular}{lcc}
\toprule
 & Ratio $\leq 1$ & Ratio $> 1$ \\
\midrule
  VaR pass (Kupiec $p > 0.05$) & """ + str(len(df_cc[(df_cc["ratio_R"] <= 1) & (df_cc["kupiec_p"] > 0.05)])) + r""" & """ + str(len(df_cc[(df_cc["ratio_R"] > 1) & (df_cc["kupiec_p"] > 0.05)])) + r""" \\
  VaR reject (Kupiec $p \leq 0.05$) & """ + str(len(df_cc[(df_cc["ratio_R"] <= 1) & (df_cc["kupiec_p"] <= 0.05)])) + r""" & """ + str(len(df_cc[(df_cc["ratio_R"] > 1) & (df_cc["kupiec_p"] <= 0.05)])) + r""" \\
\midrule
  Spearman rank correlation & \multicolumn{2}{c}{$\rho = """ + f"{rho:.3f}" + r""",\; p = """ + f"{p_spearman:.3f}" + r"""$} \\
\bottomrule
\end{tabular}"""
(OUT / "tables" / "var_diagnostic_regression.tex").write_text(tex_diag)
print(f"\nUpdated var_diagnostic_regression.tex with Spearman rho")
