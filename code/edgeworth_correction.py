"""
Edgeworth-corrected confidence interval for the ES tail average
at small effective sample sizes (nα ≈ 6).

The tail average estimator averages k ≈ nα observations from the
conditional distribution X | X ≤ VaR_α. For small k, the Gaussian
CLT approximation is inaccurate. The one-term Edgeworth expansion
corrects the quantile of the standardised mean:

    z_corrected(p) = z_p + γ₁/(6√k) · (z_p² - 1)

where γ₁ is the skewness of the conditional tail distribution.

The corrected CI half-width is:
    w_E = (σ_tail / √k) · z_corrected(1 - δ/2)

The inflation factor is:
    f(k) = w_E / w_G = z_corrected / z_{1-δ/2}
"""

import numpy as np
from scipy import stats
from scipy.integrate import quad

def conditional_tail_moments(dist, alpha):
    """Compute mean, std, skewness, kurtosis of X | X ≤ VaR_α."""
    var_alpha = dist.ppf(alpha)

    def integrand_m(x):
        return x * dist.pdf(x)
    def integrand_m2(x):
        return x**2 * dist.pdf(x)
    def integrand_m3(x):
        return x**3 * dist.pdf(x)
    def integrand_m4(x):
        return x**4 * dist.pdf(x)

    m1, _ = quad(integrand_m, -np.inf, var_alpha)
    m2, _ = quad(integrand_m2, -np.inf, var_alpha)
    m3, _ = quad(integrand_m3, -np.inf, var_alpha)
    m4, _ = quad(integrand_m4, -np.inf, var_alpha)

    # Conditional moments: divide by α
    mu = m1 / alpha
    var = m2 / alpha - mu**2
    sigma = np.sqrt(var)

    # Standardised third and fourth central moments
    cm3 = m3/alpha - 3*mu*(m2/alpha) + 2*mu**3
    gamma1 = cm3 / sigma**3  # skewness

    cm4 = m4/alpha - 4*mu*(m3/alpha) + 6*mu**2*(m2/alpha) - 3*mu**4
    gamma2 = cm4 / sigma**4 - 3  # excess kurtosis

    return mu, sigma, gamma1, gamma2


def edgeworth_inflation_factor(k, gamma1, gamma2, confidence=0.95):
    """
    Compute the Edgeworth-corrected CI inflation factor.

    Two-term Edgeworth expansion for the quantile of √k(X̄ - μ)/σ:
        z_E = z + (γ₁/6)(z² - 1)/√k + (γ₂/24)(z³ - 3z)/k
                - (γ₁²/36)(2z³ - 5z)/k

    f = z_E / z
    """
    delta = 1 - confidence
    z = stats.norm.ppf(1 - delta/2)

    # One-term (skewness only)
    z_E1 = z + (gamma1 / 6) * (z**2 - 1) / np.sqrt(k)

    # Two-term (skewness + kurtosis)
    z_E2 = (z
            + (gamma1 / 6) * (z**2 - 1) / np.sqrt(k)
            + (gamma2 / 24) * (z**3 - 3*z) / k
            - (gamma1**2 / 36) * (2*z**3 - 5*z) / k)

    f1 = z_E1 / z
    f2 = z_E2 / z

    return f1, f2, z, z_E1, z_E2


def monte_carlo_inflation(dist, alpha, n, n_sim=50000):
    """Monte Carlo estimate of the inflation factor."""
    var_alpha = dist.ppf(alpha)
    es_true = quad(lambda x: x * dist.pdf(x), -np.inf, var_alpha)[0] / alpha

    sigma_tail = conditional_tail_moments(dist, alpha)[1]
    k = n * alpha
    asymptotic_se = sigma_tail / np.sqrt(k)

    es_hats = []
    for _ in range(n_sim):
        x = dist.rvs(size=n)
        tail = x[x <= var_alpha]
        if len(tail) >= 1:
            es_hats.append(np.mean(tail))

    empirical_se = np.std(es_hats, ddof=1)
    f_mc = empirical_se / asymptotic_se

    return f_mc, empirical_se, asymptotic_se, len(es_hats)


# ═══════════════════════════════════════════════════════════════════
# Main computation
# ═══════════════════════════════════════════════════════════════════

print("=" * 72)
print("EDGEWORTH CORRECTION FOR ES TAIL AVERAGE")
print("=" * 72)

alphas = [0.01, 0.025, 0.05]
ns = [250, 500, 1000, 2000]
dist = stats.t(df=5)

print("\n--- Conditional tail distribution moments (Student-t5) ---")
for alpha in alphas:
    mu, sigma, gamma1, gamma2 = conditional_tail_moments(dist, alpha)
    print(f"  α={alpha:.3f}: μ_tail={mu:.4f}, σ_tail={sigma:.4f}, "
          f"γ₁={gamma1:.4f}, γ₂={gamma2:.4f}")

print("\n--- Edgeworth inflation factors (95% CI) ---")
print(f"{'α':>6s} {'n':>6s} {'k=nα':>6s} {'f(1-term)':>10s} "
      f"{'f(2-term)':>10s} {'z':>6s} {'z_E1':>7s} {'z_E2':>7s}")
print("-" * 65)

results = {}
for alpha in alphas:
    mu, sigma, gamma1, gamma2 = conditional_tail_moments(dist, alpha)
    for n in ns:
        k = n * alpha
        f1, f2, z, zE1, zE2 = edgeworth_inflation_factor(
            k, gamma1, gamma2, confidence=0.95)
        print(f"{alpha:>6.3f} {n:>6d} {k:>6.2f} {f1:>10.4f} "
              f"{f2:>10.4f} {z:>6.3f} {zE1:>7.3f} {zE2:>7.3f}")
        results[(alpha, n)] = (f1, f2, k, gamma1, gamma2)

print("\n--- Monte Carlo verification (50k sims, Student-t5) ---")
print(f"{'α':>6s} {'n':>6s} {'k=nα':>6s} {'f_MC':>8s} "
      f"{'f_Edge2':>8s} {'emp_SE':>10s} {'asy_SE':>10s}")
print("-" * 60)

for alpha in [0.01, 0.025, 0.05]:
    for n in [250, 500, 1000]:
        f_mc, emp_se, asy_se, n_valid = monte_carlo_inflation(
            dist, alpha, n, n_sim=20000)
        f2 = results[(alpha, n)][1]
        print(f"{alpha:>6.3f} {n:>6d} {n*alpha:>6.2f} {f_mc:>8.4f} "
              f"{f2:>8.4f} {emp_se:>10.6f} {asy_se:>10.6f}")


# Additional DGPs: skewed-t
print("\n\n--- Skewed-t(5, skew=-0.3) conditional tail moments ---")
# Use Hansen's skewed-t approximation via location-scale shift
# For a tractable approximation, use a mixture approach
# Actually scipy doesn't have skewed-t directly; use nct (noncentral t)
dist_skew = stats.nct(df=5, nc=-0.3)
for alpha in alphas:
    mu, sigma, gamma1, gamma2 = conditional_tail_moments(dist_skew, alpha)
    print(f"  α={alpha:.3f}: σ_tail={sigma:.4f}, γ₁={gamma1:.4f}, γ₂={gamma2:.4f}")
    for n in [250]:
        k = n * alpha
        f1, f2, z, zE1, zE2 = edgeworth_inflation_factor(
            k, gamma1, gamma2, confidence=0.95)
        print(f"    n={n}, k={k:.2f}: f(2-term)={f2:.4f}")


# Generate LaTeX table
print("\n\n--- LaTeX table for paper ---")
print(r"\begin{tabular}{cc rrr rr}")
print(r"\toprule")
print(r"$\alpha$ & $n$ & $k = n\alpha$ & $\gamma_1$ & $\gamma_2$ "
      r"& $f^{(1)}$ & $f^{(2)}$ \\")
print(r"\midrule")
for alpha in alphas:
    mu, sigma, gamma1, gamma2 = conditional_tail_moments(dist, alpha)
    for i, n in enumerate(ns):
        k = n * alpha
        f1, f2, z, zE1, zE2 = edgeworth_inflation_factor(
            k, gamma1, gamma2, confidence=0.95)
        alpha_str = f"${alpha*100:.1f}\\%$" if i == 0 else ""
        print(f"{alpha_str} & {n} & {k:.2f} & "
              f"{gamma1:.3f} & {gamma2:.2f} & "
              f"{f1:.3f} & {f2:.3f} \\\\")
    if alpha != alphas[-1]:
        print(r"\addlinespace")
print(r"\bottomrule")
print(r"\end{tabular}")

