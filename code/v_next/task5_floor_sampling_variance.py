"""
Monte Carlo study of the sampling distribution of the precision floor.

For each (DGP, n, alpha), draw N_SIM windows, compute:
  - C_hat = SD of tail residuals (X | X <= VaR_alpha)
  - B_hat = C_hat / sqrt(n * alpha)

Report: mean(B), SD(B), CV(B) = SD(B)/mean(B).

DGPs: Student-t(5), GARCH(1,1)-t(5).
n in {250, 500, 750, 1000}.
alpha = 0.025 (FRTB).

Generates LaTeX table and saves CSV.
"""

import numpy as np
from scipy import stats
import pandas as pd
import os

np.random.seed(2026)

N_SIM = 50_000

BASE = '/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData'
OUT = os.path.join(BASE, 'tables')

# ── DGP helpers ─────────────────────────────────────────────────

def simulate_garch_t(n_paths, path_len, omega=1e-6, alpha_g=0.10,
                     beta_g=0.85, df=5):
    paths = np.zeros((n_paths, path_len))
    sigma2 = np.full(n_paths, omega / (1 - alpha_g - beta_g))
    for t in range(path_len):
        z = stats.t.rvs(df, size=n_paths)
        z = z / np.sqrt(df / (df - 2))
        paths[:, t] = np.sqrt(sigma2) * z
        if t < path_len - 1:
            sigma2 = omega + alpha_g * paths[:, t]**2 + beta_g * sigma2
    return paths

# ── Core simulation ─────────────────────────────────────────────

def floor_sampling_variance_iid(dist, alpha, n, n_sim=N_SIM):
    """Compute B_hat = C_hat / sqrt(n*alpha) across n_sim windows."""
    var_alpha = dist.ppf(alpha)
    x = dist.rvs(size=(n_sim, n))

    c_hats = np.full(n_sim, np.nan)
    b_hats = np.full(n_sim, np.nan)
    for i in range(n_sim):
        tail = x[i, x[i] <= var_alpha]
        if len(tail) >= 2:
            c_hats[i] = np.std(tail, ddof=1)
            b_hats[i] = c_hats[i] / np.sqrt(n * alpha)

    valid = ~np.isnan(b_hats)
    c_valid = c_hats[valid]
    b_valid = b_hats[valid]

    return {
        'mean_C': np.mean(c_valid),
        'sd_C': np.std(c_valid, ddof=1),
        'cv_C': np.std(c_valid, ddof=1) / np.mean(c_valid),
        'mean_B': np.mean(b_valid),
        'sd_B': np.std(b_valid, ddof=1),
        'cv_B': np.std(b_valid, ddof=1) / np.mean(b_valid),
        'n_valid': int(valid.sum()),
        'pct5_B': np.percentile(b_valid, 5),
        'pct95_B': np.percentile(b_valid, 95),
    }

def floor_sampling_variance_garch(alpha, n, n_sim=N_SIM, burn=1000):
    """Same for GARCH(1,1)-t(5) paths."""
    total_len = burn + n
    paths = simulate_garch_t(n_sim, total_len)
    paths = paths[:, burn:]

    all_returns = paths.flatten()
    var_alpha = np.quantile(all_returns, alpha)

    c_hats = np.full(n_sim, np.nan)
    b_hats = np.full(n_sim, np.nan)
    for i in range(n_sim):
        tail = paths[i][paths[i] <= var_alpha]
        if len(tail) >= 2:
            c_hats[i] = np.std(tail, ddof=1)
            b_hats[i] = c_hats[i] / np.sqrt(n * alpha)

    valid = ~np.isnan(b_hats)
    c_valid = c_hats[valid]
    b_valid = b_hats[valid]

    return {
        'mean_C': np.mean(c_valid),
        'sd_C': np.std(c_valid, ddof=1),
        'cv_C': np.std(c_valid, ddof=1) / np.mean(c_valid),
        'mean_B': np.mean(b_valid),
        'sd_B': np.std(b_valid, ddof=1),
        'cv_B': np.std(b_valid, ddof=1) / np.mean(b_valid),
        'n_valid': int(valid.sum()),
        'pct5_B': np.percentile(b_valid, 5),
        'pct95_B': np.percentile(b_valid, 95),
    }

# ── Run ─────────────────────────────────────────────────────────

alpha = 0.025
ns = [250, 500, 750, 1000]
dist_t5 = stats.t(df=5)

rows = []
print("=" * 80)
print(f"SAMPLING VARIANCE OF PRECISION FLOOR  B = C_hat / sqrt(n*alpha)")
print(f"alpha = {alpha}, N_sim = {N_SIM:,}")
print("=" * 80)

for n in ns:
    k = n * alpha
    print(f"\n--- n = {n}, k = n*alpha = {k:.2f} ---")

    # Student-t(5)
    res_t5 = floor_sampling_variance_iid(dist_t5, alpha, n)
    print(f"  Student-t5:  mean(B)={res_t5['mean_B']:.4f}, "
          f"SD(B)={res_t5['sd_B']:.4f}, CV(B)={res_t5['cv_B']:.3f}, "
          f"90% CI=[{res_t5['pct5_B']:.4f}, {res_t5['pct95_B']:.4f}]")
    rows.append({
        'DGP': 'Student-$t_5$',
        'n': n, 'k': k,
        'mean_C': res_t5['mean_C'], 'cv_C': res_t5['cv_C'],
        'mean_B': res_t5['mean_B'], 'sd_B': res_t5['sd_B'],
        'cv_B': res_t5['cv_B'],
        'pct5_B': res_t5['pct5_B'], 'pct95_B': res_t5['pct95_B'],
    })

    # GARCH-t(5)
    res_ga = floor_sampling_variance_garch(alpha, n,
                                            n_sim=20_000 if n >= 750 else N_SIM)
    print(f"  GARCH-t5:    mean(B)={res_ga['mean_B']:.4f}, "
          f"SD(B)={res_ga['sd_B']:.4f}, CV(B)={res_ga['cv_B']:.3f}, "
          f"90% CI=[{res_ga['pct5_B']:.4f}, {res_ga['pct95_B']:.4f}]")
    rows.append({
        'DGP': 'GARCH-$t_5$',
        'n': n, 'k': k,
        'mean_C': res_ga['mean_C'], 'cv_C': res_ga['cv_C'],
        'mean_B': res_ga['mean_B'], 'sd_B': res_ga['sd_B'],
        'cv_B': res_ga['cv_B'],
        'pct5_B': res_ga['pct5_B'], 'pct95_B': res_ga['pct95_B'],
    })

df = pd.DataFrame(rows)

# ── Save CSV ────────────────────────────────────────────────────
csv_path = os.path.join(OUT, 'floor_sampling_variance.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV saved to {csv_path}")

# ── Generate LaTeX table ────────────────────────────────────────
lines = []
lines.append(r"\begin{tabular}{ll r rrrr}")
lines.append(r"\toprule")
lines.append(r"DGP & $n$ & $n\alpha$ & $\bar{B}$ & SD$(B)$ & CV$(B)$ & 90\% range \\")
lines.append(r"\midrule")

prev_dgp = None
for _, row in df.iterrows():
    dgp_str = row['DGP'] if row['DGP'] != prev_dgp else ""
    if row['DGP'] != prev_dgp and prev_dgp is not None:
        lines.append(r"\addlinespace")
    prev_dgp = row['DGP']
    rng = f"[{row['pct5_B']:.3f},\\,{row['pct95_B']:.3f}]"
    lines.append(
        f"{dgp_str} & {int(row['n'])} & {row['k']:.2f} "
        f"& {row['mean_B']:.3f} & {row['sd_B']:.3f} "
        f"& {row['cv_B']:.2f} & ${rng}$ \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

tex_content = "\n".join(lines)
tex_path = os.path.join(OUT, 'floor_sampling_variance.tex')
with open(tex_path, 'w') as f:
    f.write(tex_content)
print(f"LaTeX saved to {tex_path}")

print("\n" + tex_content)
