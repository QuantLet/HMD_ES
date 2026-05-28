"""
Task 1: Restricted pooled FE regressions for Table 4b.
Computes pooled slope b from log(SD) ~ log(n) + asset_FE + forecaster_FE
under four sample restrictions:
  1. All forecasters (reproduces existing result)
  2. Excl. TimesFM
  3. VaR-pass only (Kupiec p > 0.05 at alpha=1%)
  4. GJR + Moirai only

Deterministic seed: 42
Author: Daniel Traian Pele
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path

np.random.seed(42)

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

# Load data
ws = pd.read_csv(OUT / "tables" / "window_scaling_nonoverlap_all.csv")
recalib = pd.read_csv(OUT / "data" / "recalib_results.csv")

# Get Kupiec p-values at alpha=1% for each (asset, forecaster)
kupiec = recalib[recalib["alpha"] == 0.01][["asset", "forecaster", "kupiec_p"]].copy()


def pooled_fe_test(df_ws, label="All"):
    sub = df_ws[~df_ws["dropped"]].copy()
    if len(sub) < 5:
        return {"label": label, "cells": 0, "slope": np.nan, "se": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "p_val": np.nan}

    sub["log_n"] = np.log(sub["n"])
    sub["log_sd"] = np.log(sub["raw_sd"])

    asset_dum = pd.get_dummies(sub["asset"], drop_first=True, dtype=float)
    fcast_dum = pd.get_dummies(sub["forecaster"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(sub)), sub["log_n"].values,
                         asset_dum.values, fcast_dum.values])
    y = sub["log_sd"].values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n_obs = len(y)
    k = X.shape[1]
    mse = np.sum(resid**2) / (n_obs - k)
    cov = mse * np.linalg.inv(X.T @ X)
    se_b = np.sqrt(cov[1, 1])
    t_val = (beta[1] - (-0.5)) / se_b
    p_val = 2 * sp_stats.norm.sf(abs(t_val))

    return {
        "label": label,
        "cells": n_obs,
        "slope": beta[1],
        "se": se_b,
        "ci_lo": beta[1] - 1.96 * se_b,
        "ci_hi": beta[1] + 1.96 * se_b,
        "p_val": p_val,
    }


# --- 1. All forecasters ---
res_all = pooled_fe_test(ws, "All forecasters")

# --- 2. Excl. TimesFM ---
ws_no_tfm = ws[ws["forecaster"] != "TimesFM-2.5"]
res_no_tfm = pooled_fe_test(ws_no_tfm, "Excl.\\ TimesFM")

# --- 3. VaR-pass only (Kupiec p > 0.05) ---
var_pass = kupiec[kupiec["kupiec_p"] > 0.05][["asset", "forecaster"]]
ws_varpass = ws.merge(var_pass, on=["asset", "forecaster"], how="inner")
res_varpass = pooled_fe_test(ws_varpass, "VaR-pass only")

# --- 4. GJR + Moirai only ---
ws_gm = ws[ws["forecaster"].isin(["GJR-GARCH-t", "Moirai-2.0"])]
res_gm = pooled_fe_test(ws_gm, "GJR + Moirai only")

results = [res_all, res_no_tfm, res_varpass, res_gm]

# Print results
print("\n" + "="*75)
print("Table 4b: Restricted pooled FE rate tests")
print("="*75)
for r in results:
    print(f"\n  {r['label']}:")
    print(f"    Cells = {r['cells']}")
    print(f"    b_hat = {r['slope']:.3f}  (SE = {r['se']:.3f})")
    print(f"    95% CI = [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    print(f"    p(b = -0.5) = {r['p_val']:.3f}")

# How many VaR-pass cells?
print(f"\nVaR-pass (asset,forecaster) pairs at alpha=1%: {len(var_pass)}")
print(f"VaR-pass pairs:\n{var_pass.to_string(index=False)}")

# Generate LaTeX table
def fmt_ci(lo, hi):
    return f"$[{lo:.2f},\\;{hi:.2f}]$"

def fmt_p(p):
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"

tex_lines = [
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Sample & Cells & $\hat{b}$ & SE & 95\% CI & $p(b{=}{-}0.5)$ \\",
    r"\midrule",
]

for r in results:
    if np.isnan(r["slope"]):
        tex_lines.append(
            f"  {r['label']} & {r['cells']} & --- & --- & --- & --- \\\\"
        )
    else:
        tex_lines.append(
            f"  {r['label']} & {r['cells']} & "
            f"${r['slope']:.3f}$ & ${r['se']:.3f}$ & "
            f"{fmt_ci(r['ci_lo'], r['ci_hi'])} & {fmt_p(r['p_val'])} \\\\"
        )

tex_lines += [r"\bottomrule", r"\end{tabular}"]
tex = "\n".join(tex_lines)
(OUT / "tables" / "restricted_rate_tests.tex").write_text(tex)
print(f"\nLaTeX table saved to tables/restricted_rate_tests.tex")
print(tex)
