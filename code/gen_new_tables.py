"""Generate LaTeX tables for new analyses.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

TABDIR = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/tables")

# ── Pooled FE rate test ──────────────────────────────────────────────
ws = pd.read_csv(TABDIR / "window_scaling_nonoverlap.csv")
ws = ws[ws["dropped"] == False].copy()
ws["log_n"] = np.log(ws["n"])
ws["log_sd"] = np.log(ws["raw_sd"])

assets = ws["asset"].unique()
dummies = pd.get_dummies(ws["asset"], drop_first=True, dtype=float)
X = np.column_stack([np.ones(len(ws)), ws["log_n"].values, dummies.values])
y = ws["log_sd"].values

beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
n_obs = len(y)
k = X.shape[1]
mse = np.sum(resid**2) / (n_obs - k)
cov = mse * np.linalg.inv(X.T @ X)
se_b = np.sqrt(cov[1, 1])
t_val = (beta[1] - (-0.5)) / se_b
p_val = 2 * stats.norm.sf(abs(t_val))
r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

tex = r"""\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
  Pooled slope $\hat{b}$ & $""" + f"{beta[1]:.3f}" + r"""$ \\
  Standard error & $""" + f"{se_b:.3f}" + r"""$ \\
  95\% CI & $[""" + f"{beta[1]-1.96*se_b:.3f}" + r""",\;""" + f"{beta[1]+1.96*se_b:.3f}" + r"""]$ \\
  $t$-stat ($H_0\colon b = -0.50$) & $""" + f"{t_val:.2f}" + r"""$ \\
  $p$-value (two-sided) & $""" + f"{p_val:.3f}" + r"""$ \\
  $R^2$ & $""" + f"{r2:.3f}" + r"""$ \\
  Assets & """ + f"{len(assets)}" + r""" \\
  Observations & """ + f"{n_obs}" + r""" \\
\bottomrule
\end{tabular}"""

(TABDIR / "pooled_rate_test.tex").write_text(tex)
print(f"pooled_rate_test.tex: b={beta[1]:.3f}, SE={se_b:.3f}, t={t_val:.2f}, p={p_val:.3f}")


# ── Falsification regression ─────────────────────────────────────────
df = pd.read_csv(Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/data/recalib_results.csv"))
reg = df.copy()
reg["log_sd"] = np.log(reg["detrended_sd"])
reg["log_nalpha"] = np.log(reg["n_alpha"])
reg["log_sigma"] = np.log(reg["sigma_tail"])
reg["log_n"] = np.log(250)

y = reg["log_sd"].values
ss_tot = np.sum((y - y.mean())**2)

# Full model
X1 = np.column_stack([np.ones(len(reg)), reg["log_nalpha"].values, reg["log_sigma"].values])
b1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
r1 = y - X1 @ b1
r2_1 = 1 - np.sum(r1**2) / ss_tot
mse1 = np.sum(r1**2) / (len(y) - 3)
se1 = np.sqrt(mse1 * np.diag(np.linalg.inv(X1.T @ X1)))

# Naive: log(n) + log(sigma)
X2 = np.column_stack([np.ones(len(reg)), reg["log_n"].values, reg["log_sigma"].values])
b2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
r2_2 = 1 - np.sum((y - X2 @ b2)**2) / ss_tot
mse2 = np.sum((y - X2 @ b2)**2) / (len(y) - 3)
se2 = np.sqrt(mse2 * np.diag(np.linalg.inv(X2.T @ X2)))

# Sigma only
X3 = np.column_stack([np.ones(len(reg)), reg["log_sigma"].values])
b3, _, _, _ = np.linalg.lstsq(X3, y, rcond=None)
r2_3 = 1 - np.sum((y - X3 @ b3)**2) / ss_tot
mse3 = np.sum((y - X3 @ b3)**2) / (len(y) - 2)
se3 = np.sqrt(mse3 * np.diag(np.linalg.inv(X3.T @ X3)))

tex = r"""\begin{tabular}{lccc}
\toprule
Model & $\hat{b}$ (\textsc{se}) & $\hat{\gamma}$ (\textsc{se}) & $R^2$ \\
\midrule
  $\log(n\alpha) + \log(\hat{\sigma}_{\mathrm{tail}})$ &
    $""" + f"{b1[1]:.3f}$ $({se1[1]:.3f})$" + r""" &
    $""" + f"{b1[2]:.3f}$ $({se1[2]:.3f})$" + r""" &
    $""" + f"{r2_1:.3f}" + r"""$ \\
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
print(f"falsification_regression.tex: b={b1[1]:.3f} (SE={se1[1]:.3f}), "
      f"gamma={b1[2]:.3f}, R2_full={r2_1:.3f}, R2_naive={r2_2:.3f}")


# ── Decision experiment ──────────────────────────────────────────────
results = []
for alpha_val in [0.01, 0.025, 0.05]:
    sub = df[df["alpha"] == alpha_val]
    forecasters = sub["forecaster"].unique()
    assets = sub["asset"].unique()
    n_comp = 0
    n_frag = 0
    for asset in assets:
        ad = sub[sub["asset"] == asset]
        for i, f1 in enumerate(forecasters):
            for f2 in forecasters[i+1:]:
                r1 = ad[ad["forecaster"] == f1]
                r2 = ad[ad["forecaster"] == f2]
                if len(r1) == 0 or len(r2) == 0:
                    continue
                diff = abs(r1.iloc[0]["r_hat_mean"] - r2.iloc[0]["r_hat_mean"])
                thr = max(r1.iloc[0]["bound"], r2.iloc[0]["bound"])
                n_comp += 1
                if diff < thr:
                    n_frag += 1
    results.append((alpha_val, n_comp, n_frag))

tex = r"""\begin{tabular}{lrrr}
\toprule
$\alpha$ & Comparisons & Fragile & \% Fragile \\
\midrule"""
for a, nc, nf in results:
    pct = 100 * nf / nc
    tex += f"\n  {a*100:.1f}\\% & {nc} & {nf} & {pct:.1f}\\% \\\\"
tex += r"""
\bottomrule
\end{tabular}"""

(TABDIR / "decision_experiment.tex").write_text(tex)
print(f"decision_experiment.tex: {results}")


# ── VaR diagnostic contingency ───────────────────────────────────────
sub01 = df[df["alpha"] == 0.01].copy()
sub01["excess"] = (sub01["ratio_detrended"] > 1.0).astype(int)
sub01["kupiec_reject"] = (sub01["kupiec_p"] < 0.05).astype(int)

ct = pd.crosstab(sub01["kupiec_reject"], sub01["excess"])
vp0 = ct.loc[0, 0] if 0 in ct.index and 0 in ct.columns else 0
vp1 = ct.loc[0, 1] if 0 in ct.index and 1 in ct.columns else 0
vr0 = ct.loc[1, 0] if 1 in ct.index and 0 in ct.columns else 0
vr1 = ct.loc[1, 1] if 1 in ct.index and 1 in ct.columns else 0

odds_ratio, fisher_p = stats.fisher_exact(ct)

tex = r"""\begin{tabular}{lcc}
\toprule
 & Ratio $\leq 1$ & Ratio $> 1$ \\
\midrule"""
tex += f"\n  VaR pass (Kupiec $p > 0.05$) & {vp0} & {vp1} \\\\"
tex += f"\n  VaR reject (Kupiec $p \\leq 0.05$) & {vr0} & {vr1} \\\\"
tex += r"""
\midrule"""
if np.isinf(odds_ratio):
    tex += f"\n  Fisher exact test & \\multicolumn{{2}}{{c}}{{$p = {fisher_p:.3f}$}} \\\\"
else:
    tex += (f"\n  Fisher exact OR & \\multicolumn{{2}}{{c}}"
            f"{{${odds_ratio:.1f}$ ($p = {fisher_p:.3f}$)}} \\\\")
tex += r"""
\bottomrule
\end{tabular}"""

(TABDIR / "var_diagnostic_regression.tex").write_text(tex)
print(f"var_diagnostic_regression.tex: VaR pass {vp0}/{vp0+vp1}, "
      f"VaR reject {vr0}/{vr0+vr1}, Fisher p={fisher_p:.3f}")

print("\nAll tables generated.")
