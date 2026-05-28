"""
New empirical analyses for paper revision:
  1. Decision experiment: fragile ES model comparisons
  2. Scaling regression: log(SD) ~ log(n*alpha) + log(sigma_tail)
  3. VaR-first diagnostic regression: ratio > 1 ~ Kupiec rejection
  4. Pooled rate test from non-overlapping slopes

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
FIGDIR = OUT / "figures"
TABDIR = OUT / "tables"

MAINBLUE = "#003DA5"
IDARED   = "#C8102E"
FOREST   = "#228B22"
CRIMSON  = "#DC143C"
FCOLORS = {"GJR-GARCH-t": MAINBLUE, "TimesFM-2.5": IDARED,
           "Chronos-Small": FOREST, "Moirai-2.0": CRIMSON}
MARKERS = {"GJR-GARCH-t": "o", "TimesFM-2.5": "s",
           "Chronos-Small": "^", "Moirai-2.0": "D"}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
SAVE_KW = dict(bbox_inches="tight", transparent=True)

df = pd.read_csv(OUT / "data" / "recalib_results.csv")
roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])


# =====================================================================
# 1. DECISION EXPERIMENT: Fragile ES model comparisons
# =====================================================================
print("=" * 60)
print("1. DECISION EXPERIMENT")
print("=" * 60)

results_decision = []

for alpha_val in [0.01, 0.025, 0.05]:
    sub = df[df["alpha"] == alpha_val].copy()
    forecasters = sub["forecaster"].unique()
    assets = sub["asset"].unique()

    n_comparisons = 0
    n_fragile = 0
    fragile_details = []

    for asset in assets:
        asset_data = sub[sub["asset"] == asset]
        for i, f1 in enumerate(forecasters):
            for f2 in forecasters[i+1:]:
                row1 = asset_data[asset_data["forecaster"] == f1]
                row2 = asset_data[asset_data["forecaster"] == f2]
                if len(row1) == 0 or len(row2) == 0:
                    continue

                r1 = row1.iloc[0]["r_hat_mean"]
                r2 = row2.iloc[0]["r_hat_mean"]
                es_diff = abs(r1 - r2)

                bound1 = row1.iloc[0]["bound"]
                bound2 = row2.iloc[0]["bound"]
                threshold = max(bound1, bound2)

                n_comparisons += 1
                if es_diff < threshold:
                    n_fragile += 1
                    fragile_details.append({
                        "asset": asset,
                        "f1": f1, "f2": f2,
                        "es_diff": es_diff,
                        "threshold": threshold,
                        "ratio": es_diff / threshold
                    })

    pct = 100.0 * n_fragile / n_comparisons if n_comparisons > 0 else 0
    results_decision.append({
        "alpha": alpha_val,
        "n_comparisons": n_comparisons,
        "n_fragile": n_fragile,
        "pct_fragile": pct
    })
    print(f"  alpha={alpha_val}: {n_fragile}/{n_comparisons} ({pct:.1f}%) "
          f"pairwise ES comparisons below tolerance")

dec_df = pd.DataFrame(results_decision)
dec_df.to_csv(TABDIR / "decision_experiment.csv", index=False)

# LaTeX table
lines = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"$\alpha$ & Comparisons & Fragile & \% Fragile \\",
         r"\midrule"]
for _, row in dec_df.iterrows():
    lines.append(f"  {row['alpha']:.1%} & {int(row['n_comparisons'])} & "
                 f"{int(row['n_fragile'])} & {row['pct_fragile']:.1f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABDIR / "decision_experiment.tex").write_text("\n".join(lines))
print("  Table saved: decision_experiment.tex")


# =====================================================================
# 2. FALSIFICATION REGRESSION
# =====================================================================
print("\n" + "=" * 60)
print("2. FALSIFICATION REGRESSION")
print("=" * 60)

reg_df = df.copy()
reg_df["log_sd"] = np.log(reg_df["detrended_sd"])
reg_df["log_nalpha"] = np.log(reg_df["n_alpha"])
reg_df["log_sigma"] = np.log(reg_df["sigma_tail"])
reg_df["log_n"] = np.log(250)  # all windows are n=250

# Model 1: log(SD) = a + b*log(n*alpha) + gamma*log(sigma_tail)
from numpy.linalg import lstsq

X_full = np.column_stack([
    np.ones(len(reg_df)),
    reg_df["log_nalpha"].values,
    reg_df["log_sigma"].values
])
y = reg_df["log_sd"].values

beta_full, residuals_full, _, _ = lstsq(X_full, y, rcond=None)
y_hat_full = X_full @ beta_full
ss_res_full = np.sum((y - y_hat_full)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_full = 1 - ss_res_full / ss_tot
n_obs = len(y)
k_full = 3
se_full = np.sqrt(ss_res_full / (n_obs - k_full) * np.diag(np.linalg.inv(X_full.T @ X_full)))

print(f"  Full model: log(SD) = {beta_full[0]:.3f} + {beta_full[1]:.3f}*log(n*alpha) "
      f"+ {beta_full[2]:.3f}*log(sigma_tail)")
print(f"    R² = {r2_full:.4f}")
print(f"    b (n*alpha coeff) = {beta_full[1]:.3f} (SE = {se_full[1]:.3f}), "
      f"theory = -0.50")
print(f"    t-stat for b = -0.5: {(beta_full[1] - (-0.5))/se_full[1]:.2f}")
print(f"    gamma (sigma coeff) = {beta_full[2]:.3f} (SE = {se_full[2]:.3f}), "
      f"theory = 1.00")

# Model 2: Naive model with log(n) only (no alpha)
X_naive = np.column_stack([
    np.ones(len(reg_df)),
    reg_df["log_n"].values,
    reg_df["log_sigma"].values
])
beta_naive, _, _, _ = lstsq(X_naive, y, rcond=None)
y_hat_naive = X_naive @ beta_naive
ss_res_naive = np.sum((y - y_hat_naive)**2)
r2_naive = 1 - ss_res_naive / ss_tot

print(f"\n  Naive model (log(n) instead of log(n*alpha)): R² = {r2_naive:.4f}")
print(f"    Delta R² = {r2_full - r2_naive:.4f}")

# Model 3: Unconditional SD only
X_sigma = np.column_stack([
    np.ones(len(reg_df)),
    reg_df["log_sigma"].values
])
beta_sigma, _, _, _ = lstsq(X_sigma, y, rcond=None)
y_hat_sigma = X_sigma @ beta_sigma
ss_res_sigma = np.sum((y - y_hat_sigma)**2)
r2_sigma = 1 - ss_res_sigma / ss_tot

print(f"  Sigma-only model: R² = {r2_sigma:.4f}")
print(f"    Delta R² vs full = {r2_full - r2_sigma:.4f}")

# Save regression results
reg_results = {
    "Model": ["Full: log(n*alpha) + log(sigma)",
              "Naive: log(n) + log(sigma)",
              "Sigma only: log(sigma)"],
    "b_nalpha": [f"{beta_full[1]:.3f} ({se_full[1]:.3f})", "---", "---"],
    "gamma_sigma": [f"{beta_full[2]:.3f} ({se_full[2]:.3f})",
                    f"{beta_naive[2]:.3f}", f"{beta_sigma[1]:.3f}"],
    "R2": [f"{r2_full:.4f}", f"{r2_naive:.4f}", f"{r2_sigma:.4f}"]
}

# LaTeX table for falsification
lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Model & $\hat{b}$ (\textsc{se}) & $\hat{\gamma}$ (\textsc{se}) & $R^2$ \\",
         r"\midrule"]
lines.append(f"  $\\log(n\\alpha) + \\log(\\hat{{\\sigma}}_{{\\mathrm{{tail}}}})$ & "
             f"${beta_full[1]:.3f}$ $({se_full[1]:.3f})$ & "
             f"${beta_full[2]:.3f}$ $({se_full[2]:.3f})$ & "
             f"${r2_full:.3f}$ \\\\")
lines.append(f"  $\\log(n) + \\log(\\hat{{\\sigma}}_{{\\mathrm{{tail}}}})$ & "
             f"--- & ${beta_naive[2]:.3f}$ & ${r2_naive:.3f}$ \\\\")
lines.append(f"  $\\log(\\hat{{\\sigma}}_{{\\mathrm{{tail}}}})$ only & "
             f"--- & ${beta_sigma[1]:.3f}$ & ${r2_sigma:.3f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABDIR / "falsification_regression.tex").write_text("\n".join(lines))
print("  Table saved: falsification_regression.tex")


# =====================================================================
# 3. VAR-FIRST DIAGNOSTIC REGRESSION
# =====================================================================
print("\n" + "=" * 60)
print("3. VAR-FIRST DIAGNOSTIC REGRESSION")
print("=" * 60)

sub01 = df[df["alpha"] == 0.01].copy()

# Binary: does excess dispersion (ratio > 1) associate with VaR rejection?
sub01["excess"] = (sub01["ratio_detrended"] > 1.0).astype(int)
sub01["kupiec_reject"] = (sub01["kupiec_p"] < 0.05).astype(int)

# Contingency table
ct = pd.crosstab(sub01["kupiec_reject"], sub01["excess"],
                 margins=True, margins_name="Total")
ct.index = ct.index.map({0: "VaR pass", 1: "VaR reject", "Total": "Total"})
ct.columns = ct.columns.map({0: "Ratio ≤ 1", 1: "Ratio > 1", "Total": "Total"})
print("  Contingency table (alpha=1%):")
print(ct)

# Fisher exact test
table_2x2 = pd.crosstab(sub01["kupiec_reject"], sub01["excess"])
odds_ratio, fisher_p = stats.fisher_exact(table_2x2)
print(f"\n  Fisher exact test: OR = {odds_ratio:.2f}, p = {fisher_p:.4e}")

# Logistic regression: P(ratio > 1) ~ kupiec_reject
from scipy.optimize import minimize

def neg_loglik(beta, X, y):
    z = X @ beta
    z = np.clip(z, -30, 30)
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

X_logit = np.column_stack([np.ones(len(sub01)), sub01["kupiec_reject"].values])
y_logit = sub01["excess"].values

res = minimize(neg_loglik, [0, 0], args=(X_logit, y_logit), method="Nelder-Mead")
beta_logit = res.x
print(f"\n  Logistic regression: P(ratio > 1) = logit(b0 + b1*VaR_reject)")
print(f"    b0 = {beta_logit[0]:.3f}, b1 = {beta_logit[1]:.3f}")
print(f"    P(excess | VaR pass) = {1/(1+np.exp(-beta_logit[0])):.3f}")
print(f"    P(excess | VaR reject) = {1/(1+np.exp(-(beta_logit[0]+beta_logit[1]))):.3f}")

# Continuous regression: log(ratio) ~ -log(kupiec_p)
sub01_valid = sub01[sub01["kupiec_p"] > 0].copy()
sub01_valid["log_ratio"] = np.log(sub01_valid["ratio_detrended"])
sub01_valid["neg_log_p"] = -np.log(sub01_valid["kupiec_p"])

slope_vf, intercept_vf, r_vf, p_vf, se_vf = stats.linregress(
    sub01_valid["neg_log_p"], sub01_valid["log_ratio"])
print(f"\n  Continuous regression: log(ratio) = {intercept_vf:.3f} + "
      f"{slope_vf:.4f}*(-log(p_Kupiec))")
print(f"    R² = {r_vf**2:.4f}, p = {p_vf:.4e}")

# Also across all alpha levels
sub_all = df.copy()
sub_all["excess"] = (sub_all["ratio_detrended"] > 1.0).astype(int)
sub_all["kupiec_reject"] = (sub_all["kupiec_p"] < 0.05).astype(int)
n_excess_total = sub_all["excess"].sum()
n_total = len(sub_all)
n_excess_reject = sub_all[sub_all["kupiec_reject"] == 1]["excess"].sum()
n_reject = sub_all["kupiec_reject"].sum()
print(f"\n  All alphas: {n_excess_total}/{n_total} ({100*n_excess_total/n_total:.1f}%) "
      f"have ratio > 1")
print(f"  Among VaR rejects: {n_excess_reject}/{n_reject} ({100*n_excess_reject/n_reject:.1f}%)")
n_excess_pass = sub_all[sub_all["kupiec_reject"] == 0]["excess"].sum()
n_pass = n_total - n_reject
print(f"  Among VaR passes: {n_excess_pass}/{n_pass} ({100*n_excess_pass/n_pass:.1f}%)")

# Save LaTeX table for diagnostic regression
lines = [r"\begin{tabular}{lcc}", r"\toprule",
         r" & Ratio $\leq 1$ & Ratio $> 1$ \\",
         r"\midrule"]
ct_raw = pd.crosstab(sub01["kupiec_reject"], sub01["excess"])
vp_le = ct_raw.loc[0, 0] if 0 in ct_raw.index else 0
vp_gt = ct_raw.loc[0, 1] if 0 in ct_raw.index and 1 in ct_raw.columns else 0
vr_le = ct_raw.loc[1, 0] if 1 in ct_raw.index else 0
vr_gt = ct_raw.loc[1, 1] if 1 in ct_raw.index and 1 in ct_raw.columns else 0
lines.append(f"  VaR pass (Kupiec $p > 0.05$) & {vp_le} & {vp_gt} \\\\")
lines.append(f"  VaR reject (Kupiec $p \\leq 0.05$) & {vr_le} & {vr_gt} \\\\")
lines += [r"\midrule"]
lines.append(f"  Fisher exact OR & \\multicolumn{{2}}{{c}}"
             f"{{{odds_ratio:.1f} ($p = {fisher_p:.3e}$)}} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABDIR / "var_diagnostic_regression.tex").write_text("\n".join(lines))
print("  Table saved: var_diagnostic_regression.tex")


# =====================================================================
# 4. POOLED RATE TEST
# =====================================================================
print("\n" + "=" * 60)
print("4. POOLED RATE TEST")
print("=" * 60)

# Read non-overlapping slope data
slopes_df = pd.read_csv(TABDIR / "window_scaling_nonoverlap_slopes.csv")
valid_slopes = slopes_df[slopes_df["drop_reason"].isna()].copy()

print(f"  {len(valid_slopes)} assets with valid non-overlapping slopes")
print(f"  Mean slope: {valid_slopes['slope'].mean():.3f}")
print(f"  Median slope: {valid_slopes['slope'].median():.3f}")
print(f"  Weighted mean (by 1/SE²): ", end="")

weights = 1.0 / valid_slopes["se"].values**2
weighted_mean = np.average(valid_slopes["slope"].values, weights=weights)
weighted_se = 1.0 / np.sqrt(weights.sum())
t_stat = (weighted_mean - (-0.5)) / weighted_se
print(f"{weighted_mean:.3f} (SE = {weighted_se:.4f})")
print(f"  t-stat for H0: slope = -0.50: {t_stat:.2f}")
print(f"  p-value (two-sided): {2*stats.norm.sf(abs(t_stat)):.4f}")

# Also read the full window-scaling data to build a pooled regression
ws_data = pd.read_csv(TABDIR / "window_scaling_nonoverlap.csv")
print(f"\n  Window-scaling data: {len(ws_data)} rows")
print(f"  Columns: {list(ws_data.columns)}")
print(ws_data.head())

# If the non-overlap data has per-window-length SDs, do a pooled regression
if "n" in ws_data.columns and "sd" in ws_data.columns:
    ws_data["log_n"] = np.log(ws_data["n"])
    ws_data["log_sd"] = np.log(ws_data["sd"])

    # Pooled OLS with asset fixed effects
    assets_ws = ws_data["asset"].unique()
    dummies = pd.get_dummies(ws_data["asset"], drop_first=True, dtype=float)
    X_pooled = np.column_stack([
        np.ones(len(ws_data)),
        ws_data["log_n"].values,
        dummies.values
    ])
    y_pooled = ws_data["log_sd"].values

    beta_pooled, _, _, _ = np.linalg.lstsq(X_pooled, y_pooled, rcond=None)
    y_hat_pooled = X_pooled @ beta_pooled
    resid_pooled = y_pooled - y_hat_pooled
    ss_res_pooled = np.sum(resid_pooled**2)
    n_p = len(y_pooled)
    k_p = X_pooled.shape[1]
    mse_pooled = ss_res_pooled / (n_p - k_p)
    cov_pooled = mse_pooled * np.linalg.inv(X_pooled.T @ X_pooled)
    se_slope_pooled = np.sqrt(cov_pooled[1, 1])

    print(f"\n  Pooled FE regression: log(SD) = alpha_i + b*log(n)")
    print(f"    b = {beta_pooled[1]:.4f} (SE = {se_slope_pooled:.4f})")
    print(f"    t-stat for b = -0.50: {(beta_pooled[1] - (-0.5))/se_slope_pooled:.2f}")
    print(f"    95% CI: [{beta_pooled[1] - 1.96*se_slope_pooled:.4f}, "
          f"{beta_pooled[1] + 1.96*se_slope_pooled:.4f}]")

    # Save pooled results
    pooled_results = {
        "slope": beta_pooled[1],
        "se": se_slope_pooled,
        "t_vs_half": (beta_pooled[1] - (-0.5)) / se_slope_pooled,
        "ci_lo": beta_pooled[1] - 1.96*se_slope_pooled,
        "ci_hi": beta_pooled[1] + 1.96*se_slope_pooled,
        "n_assets": len(assets_ws),
        "n_obs": n_p
    }

    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Statistic & Value \\", r"\midrule"]
    lines.append(f"  Pooled slope $\\hat{{b}}$ & ${beta_pooled[1]:.3f}$ \\\\")
    lines.append(f"  Standard error & ${se_slope_pooled:.3f}$ \\\\")
    lines.append(f"  95\\% CI & $[{beta_pooled[1] - 1.96*se_slope_pooled:.3f},\\;"
                 f"{beta_pooled[1] + 1.96*se_slope_pooled:.3f}]$ \\\\")
    t_val = (beta_pooled[1] - (-0.5)) / se_slope_pooled
    lines.append(f"  $t$-stat ($H_0\\colon b = -0.50$) & ${t_val:.2f}$ \\\\")
    lines.append(f"  Assets & ${len(assets_ws)}$ \\\\")
    lines.append(f"  Observations & ${n_p}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABDIR / "pooled_rate_test.tex").write_text("\n".join(lines))
    print("  Table saved: pooled_rate_test.tex")
else:
    print("  [Warning] window_scaling_nonoverlap.csv missing n/sd columns, "
          "using inverse-variance weighted mean only")

    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Statistic & Value \\", r"\midrule"]
    lines.append(f"  Weighted mean slope & ${weighted_mean:.3f}$ \\\\")
    lines.append(f"  Weighted SE & ${weighted_se:.4f}$ \\\\")
    lines.append(f"  $t$-stat ($H_0\\colon b = -0.50$) & ${t_stat:.2f}$ \\\\")
    p_val = 2*stats.norm.sf(abs(t_stat))
    lines.append(f"  $p$-value (two-sided) & ${p_val:.4f}$ \\\\")
    lines.append(f"  Assets & ${len(valid_slopes)}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABDIR / "pooled_rate_test.tex").write_text("\n".join(lines))
    print("  Table saved: pooled_rate_test.tex")


print("\n" + "=" * 60)
print("ALL NEW ANALYSES COMPLETE")
print("=" * 60)
