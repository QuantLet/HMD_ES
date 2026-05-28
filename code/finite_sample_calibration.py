"""
Finite-sample calibration of C/√(nα) at FRTB parameters.

Three DGPs: Student-t(5), noncentral-t(5, nc=-0.3), GARCH(1,1)-t(5).
Reports inflation factor f(n,α) = empirical_SD / asymptotic_SD.

Also computes the analytical correction from random tail count:
  f_analytical = √(1 + (1-α)/(nα))
"""

import numpy as np
from scipy import stats
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

N_SIM = 50_000

def conditional_tail_sigma(dist, alpha):
    """σ_tail = SD(X | X ≤ VaR_α)."""
    var_alpha = dist.ppf(alpha)
    def m1_int(x): return x * dist.pdf(x)
    def m2_int(x): return x**2 * dist.pdf(x)
    m1, _ = quad(m1_int, -np.inf, var_alpha)
    m2, _ = quad(m2_int, -np.inf, var_alpha)
    mu = m1 / alpha
    var = m2 / alpha - mu**2
    return np.sqrt(var), mu

def mc_inflation_iid(dist, alpha, n, n_sim=N_SIM):
    """MC inflation factor for i.i.d. draws from dist."""
    var_alpha = dist.ppf(alpha)
    sigma_tail, es_true = conditional_tail_sigma(dist, alpha)
    asymp_se = sigma_tail / np.sqrt(n * alpha)

    x = dist.rvs(size=(n_sim, n))
    es_hats = []
    for i in range(n_sim):
        tail = x[i, x[i] <= var_alpha]
        if len(tail) >= 1:
            es_hats.append(np.mean(tail))
    emp_se = np.std(es_hats, ddof=1)
    return emp_se / asymp_se, emp_se, asymp_se, sigma_tail

def simulate_garch_t(n_paths, path_len, omega=1e-6, alpha_g=0.10,
                     beta_g=0.85, df=5):
    """Simulate GARCH(1,1)-t(df) paths."""
    paths = np.zeros((n_paths, path_len))
    sigma2 = np.full(n_paths, omega / (1 - alpha_g - beta_g))
    for t in range(path_len):
        z = stats.t.rvs(df, size=n_paths)
        # Scale t innovations to unit variance: Var(t_df) = df/(df-2)
        z = z / np.sqrt(df / (df - 2))
        paths[:, t] = np.sqrt(sigma2) * z
        if t < path_len - 1:
            sigma2 = omega + alpha_g * paths[:, t]**2 + beta_g * sigma2
    return paths

def mc_inflation_garch(alpha, n, n_sim=N_SIM, burn=1000):
    """MC inflation factor for GARCH-t(5)."""
    total_len = burn + n
    paths = simulate_garch_t(n_sim, total_len)
    paths = paths[:, burn:]  # discard burn-in

    # Use true unconditional VaR (from large MC)
    all_returns = paths.flatten()
    var_alpha = np.quantile(all_returns, alpha)

    # Compute σ_tail from MC
    tail_obs = all_returns[all_returns <= var_alpha]
    sigma_tail = np.std(tail_obs, ddof=1)
    asymp_se = sigma_tail / np.sqrt(n * alpha)

    es_hats = []
    for i in range(n_sim):
        tail = paths[i][paths[i] <= var_alpha]
        if len(tail) >= 1:
            es_hats.append(np.mean(tail))
    emp_se = np.std(es_hats, ddof=1)
    return emp_se / asymp_se, emp_se, asymp_se, sigma_tail


# ═══════════════════════════════════════════════════════════════
# Main computation
# ═══════════════════════════════════════════════════════════════

alphas = [0.01, 0.025, 0.05]
ns = [250, 500, 1000, 2000]

print("=" * 78)
print("FINITE-SAMPLE INFLATION FACTORS  f(n,α) = empirical SD / asymptotic SD")
print(f"N_sim = {N_SIM:,}")
print("=" * 78)

# Analytical correction
print("\n--- Analytical correction: f = √(1 + (1-α)/(nα)) ---")
for alpha in alphas:
    for n in ns:
        k = n * alpha
        f_anal = np.sqrt(1 + (1 - alpha) / k)
        print(f"  α={alpha:.3f}, n={n:5d}, k={k:6.2f}: f_anal = {f_anal:.4f}")

# DGP 1: Student-t(5)
print("\n--- DGP 1: Student-t(5) ---")
dist_t5 = stats.t(df=5)
results_t5 = {}
for alpha in alphas:
    for n in ns:
        f, emp, asy, stail = mc_inflation_iid(dist_t5, alpha, n)
        results_t5[(alpha, n)] = f
        print(f"  α={alpha:.3f}, n={n:5d}, k={n*alpha:6.2f}: "
              f"f={f:.4f}, σ_tail={stail:.4f}")

# DGP 2: Noncentral-t(5, nc=-0.3) [skewed-t proxy]
print("\n--- DGP 2: Noncentral-t(5, nc=-0.3) ---")
dist_nct = stats.nct(df=5, nc=-0.3)
results_nct = {}
for alpha in alphas:
    for n in ns:
        f, emp, asy, stail = mc_inflation_iid(dist_nct, alpha, n)
        results_nct[(alpha, n)] = f
        print(f"  α={alpha:.3f}, n={n:5d}, k={n*alpha:6.2f}: "
              f"f={f:.4f}, σ_tail={stail:.4f}")

# DGP 3: GARCH(1,1)-t(5)
print("\n--- DGP 3: GARCH(1,1)-t(5) ---")
results_garch = {}
for alpha in alphas:
    for n in ns:
        f, emp, asy, stail = mc_inflation_garch(alpha, n, n_sim=20000)
        results_garch[(alpha, n)] = f
        print(f"  α={alpha:.3f}, n={n:5d}, k={n*alpha:6.2f}: "
              f"f={f:.4f}, σ_tail={stail:.4f}")

# Generate LaTeX table
print("\n\n" + "=" * 78)
print("LaTeX TABLE: Finite-sample inflation factors")
print("=" * 78)

lines = []
lines.append(r"\begin{tabular}{cc r ccc}")
lines.append(r"\toprule")
lines.append(r"$\alpha$ & $n$ & $k = n\alpha$ & Student-$t_5$ "
             r"& Skewed-$t_5$ & GARCH-$t_5$ \\")
lines.append(r"\midrule")
for alpha in alphas:
    for i, n in enumerate(ns):
        k = n * alpha
        f_t5 = results_t5[(alpha, n)]
        f_nct = results_nct[(alpha, n)]
        f_ga = results_garch.get((alpha, n), float('nan'))
        alpha_str = f"${alpha*100:.1f}\\%$" if i == 0 else ""

        def flag(f):
            s = f"{f:.3f}"
            if f > 1.20:
                return s + "$^\\dagger$"
            return s

        lines.append(f"{alpha_str} & {n} & {k:.2f} & "
                      f"{flag(f_t5)} & {flag(f_nct)} & {flag(f_ga)} \\\\")
    if alpha != alphas[-1]:
        lines.append(r"\addlinespace")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
print("\n".join(lines))

# Summary statistics for paper text
print("\n\n--- Key numbers for paper prose ---")
for alpha in [0.025, 0.01]:
    n = 250
    k = n * alpha
    f_t5 = results_t5[(alpha, n)]
    f_nct = results_nct[(alpha, n)]
    f_ga = results_garch.get((alpha, n), float('nan'))
    f_anal = np.sqrt(1 + (1 - alpha) / k)
    print(f"α={alpha}, n={n}, k={k:.2f}:")
    print(f"  f_analytical = {f_anal:.3f}")
    print(f"  f_t5 = {f_t5:.3f}, f_nct = {f_nct:.3f}, f_garch = {f_ga:.3f}")
    print(f"  Max across DGPs: {max(f_t5, f_nct, f_ga):.3f}")

