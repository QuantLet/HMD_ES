"""
VaR-first diagnostic robustness: CC statistics at alpha=1%, 2.5%, 5%.
Produces a summary table of Spearman rho between CC and benchmark ratio R
at each diagnostic alpha level.

Author: Daniel Traian Pele
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data")
OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
from scipy.stats import t as student_t

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

ALPHAS = [0.01, 0.025, 0.05]


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


def christoffersen_cc(hits, alpha):
    n = len(hits)
    h = hits.astype(int)
    n1 = h.sum()
    n0 = n - n1

    pi_hat = n1 / n
    if pi_hat == 0 or pi_hat == 1:
        return np.nan, np.nan, np.nan, np.nan

    uc_stat = -2 * (n0 * np.log(1 - alpha) + n1 * np.log(alpha)
                     - n0 * np.log(1 - pi_hat) - n1 * np.log(pi_hat))

    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if h[i-1] == 0 and h[i] == 0: n00 += 1
        elif h[i-1] == 0 and h[i] == 1: n01 += 1
        elif h[i-1] == 1 and h[i] == 0: n10 += 1
        else: n11 += 1

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


recalib = pd.read_csv(OUT / "data" / "recalib_results.csv")

summary_rows = []

for alpha in ALPHAS:
    results = []
    for asset in ASSETS:
        try:
            ret = load_returns(asset)
        except Exception:
            continue

        for fname, fkey in FORECASTERS.items():
            try:
                var_hat = load_forecasts(asset, fkey, alpha)
            except Exception:
                continue

            idx = ret.index.intersection(var_hat.index).sort_values()
            X = ret.loc[idx].values
            V = var_hat.loc[idx].values
            hits = X <= V

            cc_stat, cc_pval, uc_stat, ind_stat = christoffersen_cc(hits, alpha)

            rc = recalib[(recalib["asset"] == asset) &
                         (recalib["forecaster"] == fname) &
                         (recalib["alpha"] == alpha)]
            R = rc["ratio_detrended"].values[0] if len(rc) > 0 else np.nan

            results.append({
                "asset": asset, "forecaster": fname,
                "cc_stat": cc_stat, "ratio_R": R,
            })

    df_cc = pd.DataFrame(results)
    valid = df_cc.dropna(subset=["cc_stat", "ratio_R"])

    # Full sample
    rho_full, p_full = spearmanr(valid["cc_stat"], valid["ratio_R"])

    # Excluding Chronos-Small
    excl = valid[valid["forecaster"] != "Chronos-Small"]
    if len(excl) > 5:
        rho_excl, p_excl = spearmanr(excl["cc_stat"], excl["ratio_R"])
    else:
        rho_excl, p_excl = np.nan, np.nan

    n_full = len(valid)
    n_excl = len(excl)

    summary_rows.append({
        "alpha": alpha,
        "n_pairs": n_full,
        "rho_full": rho_full,
        "p_full": p_full,
        "n_excl": n_excl,
        "rho_excl": rho_excl,
        "p_excl": p_excl,
    })

    pct = f"{alpha*100:.1f}" if alpha != 0.01 else "1"
    print(f"alpha={pct}%: n={n_full}, rho={rho_full:.3f} (p={p_full:.4f}), "
          f"excl Chronos: rho={rho_excl:.3f} (p={p_excl:.4f})")

# Generate LaTeX table
tex_lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Diagnostic $\alpha$ & $n$ pairs & Spearman $\rho$ (all) & Spearman $\rho$ (excl.\ Chronos) & Comment \\",
    r"\midrule",
]

comments = {
    0.01: "FRTB VaR gatekeeper",
    0.025: "Matched to FRTB ES level",
    0.05: "Scaling validation",
}

for row in summary_rows:
    a = row["alpha"]
    pct = f"{a*100:.1f}\\%" if a != 0.01 else "1\\%"

    def fmt_rho(rho, p):
        stars = ""
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        return f"{rho:.3f}{stars}"

    tex_lines.append(
        f"  {pct} & {row['n_pairs']} & "
        f"${fmt_rho(row['rho_full'], row['p_full'])}$ & "
        f"${fmt_rho(row['rho_excl'], row['p_excl'])}$ & "
        f"{comments[a]} \\\\"
    )

tex_lines += [r"\bottomrule", r"\end{tabular}"]
tex = "\n".join(tex_lines)
(OUT / "tables" / "var_diagnostic_robustness.tex").write_text(tex)
print(f"\nSaved to tables/var_diagnostic_robustness.tex")
