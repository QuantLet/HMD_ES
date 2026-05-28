"""
Generate revised LaTeX tables with clustered SEs and sqrt(C1^2+C2^2) threshold.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
TABDIR = OUT / "tables"

df = pd.read_csv(OUT / "data" / "recalib_results.csv")

# =====================================================================
# 1. SCALING REGRESSION with clustered SEs
# =====================================================================
reg = df.copy()
reg["log_sd"] = np.log(reg["detrended_sd"])
reg["log_nalpha"] = np.log(reg["n_alpha"])
reg["log_sigma"] = np.log(reg["sigma_tail"])
reg["log_n"] = np.log(250)

y = reg["log_sd"].values
ss_tot = np.sum((y - y.mean())**2)

# --- Full model: log(SD) = a + b*log(n*alpha) + gamma*log(sigma) ---
X1 = np.column_stack([np.ones(len(reg)), reg["log_nalpha"].values, reg["log_sigma"].values])
b1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
r1 = y - X1 @ b1
r2_1 = 1 - np.sum(r1**2) / ss_tot

# Clustered SEs by asset
assets = reg["asset"].values
unique_assets = np.unique(assets)
n_clusters = len(unique_assets)
k = X1.shape[1]

# Cluster-robust variance (CR1)
bread = np.linalg.inv(X1.T @ X1)
meat = np.zeros((k, k))
for asset in unique_assets:
    mask = assets == asset
    Xi = X1[mask]
    ei = r1[mask]
    score = Xi.T @ np.diag(ei)
    meat += score @ score.T
# Small-sample correction: G/(G-1) * (n-1)/(n-k)
n_obs = len(y)
correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k))
V_clustered = correction * bread @ meat @ bread
se_clustered = np.sqrt(np.diag(V_clustered))

# Also with forecaster FEs
forecasters = reg["forecaster"].values
f_dummies = pd.get_dummies(pd.Series(forecasters), drop_first=True, dtype=float)
X1_fe = np.column_stack([X1, f_dummies.values])
b1_fe, _, _, _ = np.linalg.lstsq(X1_fe, y, rcond=None)
r1_fe = y - X1_fe @ b1_fe
r2_1_fe = 1 - np.sum(r1_fe**2) / ss_tot

# Clustered SEs for FE model
k_fe = X1_fe.shape[1]
bread_fe = np.linalg.inv(X1_fe.T @ X1_fe)
meat_fe = np.zeros((k_fe, k_fe))
for asset in unique_assets:
    mask = assets == asset
    Xi = X1_fe[mask]
    ei = r1_fe[mask]
    score = Xi.T @ np.diag(ei)
    meat_fe += score @ score.T
correction_fe = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k_fe))
V_fe = correction_fe * bread_fe @ meat_fe @ bread_fe
se_fe = np.sqrt(np.diag(V_fe))

print("SCALING REGRESSION")
print(f"  Full model (OLS):  b={b1[1]:.3f} (SE_OLS={np.sqrt(np.sum(r1**2)/(n_obs-k) * np.diag(np.linalg.inv(X1.T @ X1))[1]):.3f})")
print(f"  Full model (CLU):  b={b1[1]:.3f} (SE_clustered={se_clustered[1]:.3f})")
print(f"    gamma={b1[2]:.3f} (SE_clustered={se_clustered[2]:.3f})")
print(f"    R2={r2_1:.3f}")
t_b = (b1[1] - (-0.5)) / se_clustered[1]
print(f"    t-stat for b=-0.5: {t_b:.2f}")
print(f"  With forecaster FE: b={b1_fe[1]:.3f} (SE_clustered={se_fe[1]:.3f}), R2={r2_1_fe:.3f}")

# --- Naive: log(n) + log(sigma) ---
X2 = np.column_stack([np.ones(len(reg)), reg["log_n"].values, reg["log_sigma"].values])
b2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
r2_2 = 1 - np.sum((y - X2 @ b2)**2) / ss_tot
r2_res = y - X2 @ b2
k2 = 3
bread2 = np.linalg.inv(X2.T @ X2)
meat2 = np.zeros((k2, k2))
for asset in unique_assets:
    mask = assets == asset
    Xi = X2[mask]
    ei = r2_res[mask]
    score = Xi.T @ np.diag(ei)
    meat2 += score @ score.T
correction2 = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k2))
V2 = correction2 * bread2 @ meat2 @ bread2
se2 = np.sqrt(np.diag(V2))

# --- Sigma only ---
X3 = np.column_stack([np.ones(len(reg)), reg["log_sigma"].values])
b3, _, _, _ = np.linalg.lstsq(X3, y, rcond=None)
r2_3 = 1 - np.sum((y - X3 @ b3)**2) / ss_tot
r3_res = y - X3 @ b3
k3 = 2
bread3 = np.linalg.inv(X3.T @ X3)
meat3 = np.zeros((k3, k3))
for asset in unique_assets:
    mask = assets == asset
    Xi = X3[mask]
    ei = r3_res[mask]
    score = Xi.T @ np.diag(ei)
    meat3 += score @ score.T
correction3 = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k3))
V3 = correction3 * bread3 @ meat3 @ bread3
se3 = np.sqrt(np.diag(V3))

# Write LaTeX table
tex = r"""\begin{tabular}{lccc}
\toprule
Model & $\hat{b}$ (\textsc{se}) & $\hat{\gamma}$ (\textsc{se}) & $R^2$ \\
\midrule
  $\log(n\alpha) + \log(\hat{\sigma}_{\mathrm{tail}})$ &
    $""" + f"{b1[1]:.3f}$ $({se_clustered[1]:.3f})$" + r""" &
    $""" + f"{b1[2]:.3f}$ $({se_clustered[2]:.3f})$" + r""" &
    $""" + f"{r2_1:.3f}" + r"""$ \\
  \quad + forecaster FE &
    $""" + f"{b1_fe[1]:.3f}$ $({se_fe[1]:.3f})$" + r""" &
    $""" + f"{b1_fe[2]:.3f}$ $({se_fe[2]:.3f})$" + r""" &
    $""" + f"{r2_1_fe:.3f}" + r"""$ \\
  $\log(n) + \log(\hat{\sigma}_{\mathrm{tail}})$ &
    --- &
    $""" + f"{b2[2]:.3f}$ $({se2[2]:.3f})$" + r""" &
    $""" + f"{r2_2:.3f}" + r"""$ \\
  $\log(\hat{\sigma}_{\mathrm{tail}})$ only &
    --- &
    $""" + f"{b3[1]:.3f}$ $({se3[1]:.3f})$" + r""" &
    $""" + f"{r2_3:.3f}" + r"""$ \\
\bottomrule
\end{tabular}"""

(TABDIR / "falsification_regression.tex").write_text(tex)
print(f"  falsification_regression.tex saved")


# =====================================================================
# 2. DECISION EXPERIMENT with sqrt(C1^2 + C2^2) threshold
# =====================================================================
print("\nDECISION EXPERIMENT (sqrt threshold)")

results = []
for alpha_val in [0.01, 0.025, 0.05]:
    sub = df[df["alpha"] == alpha_val]
    forecasters = sub["forecaster"].unique()
    assets_list = sub["asset"].unique()
    n_comp = 0
    n_frag = 0
    for asset in assets_list:
        ad = sub[sub["asset"] == asset]
        for i, f1 in enumerate(forecasters):
            for f2 in forecasters[i+1:]:
                row1 = ad[ad["forecaster"] == f1]
                row2 = ad[ad["forecaster"] == f2]
                if len(row1) == 0 or len(row2) == 0:
                    continue
                diff = abs(row1.iloc[0]["r_hat_mean"] - row2.iloc[0]["r_hat_mean"])
                c1 = row1.iloc[0]["bound"]
                c2 = row2.iloc[0]["bound"]
                threshold = np.sqrt(c1**2 + c2**2)
                n_comp += 1
                if diff < threshold:
                    n_frag += 1
    pct = 100 * n_frag / n_comp
    results.append((alpha_val, n_comp, n_frag, pct))
    print(f"  alpha={alpha_val}: {n_frag}/{n_comp} ({pct:.1f}%) fragile")

tex = r"""\begin{tabular}{lrrr}
\toprule
$\alpha$ & Comparisons & Fragile & \% Fragile \\
\midrule"""
for a, nc, nf, pct in results:
    tex += f"\n  {a*100:.1f}\\% & {nc} & {nf} & {pct:.1f}\\% \\\\"
tex += r"""
\bottomrule
\end{tabular}"""

(TABDIR / "decision_experiment.tex").write_text(tex)
print(f"  decision_experiment.tex saved")

print("\nAll tables regenerated.")
