"""
Task 4: Re-run §6 cross-sectional scaling regression with rolling σ_tail.

Tests whether γ̂ moves toward 1 when σ_tail is estimated on rolling
250-day windows rather than once over the full evaluation sample.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
recalib = pd.read_csv('/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/data/recalib_results.csv')
rolling = pd.read_csv('/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/data/rolling_estimates.csv')

# Compute rolling σ_tail median per (asset, forecaster, alpha)
roll_sigma = (rolling[rolling['sigma_tail_w'].notna()]
              .groupby(['asset', 'forecaster', 'alpha'])['sigma_tail_w']
              .median()
              .reset_index()
              .rename(columns={'sigma_tail_w': 'sigma_tail_rolling'}))

# Merge
df = recalib.merge(roll_sigma, on=['asset', 'forecaster', 'alpha'], how='left')

# Filter to cells with both
df = df[df['sigma_tail_rolling'].notna() & (df['sigma_tail_rolling'] > 0)].copy()

# Compute regression variables
df['log_sd'] = np.log(df['detrended_sd'])
df['log_nalpha'] = np.log(df['n_alpha'])
df['log_sigma_full'] = np.log(df['sigma_tail'])
df['log_sigma_roll'] = np.log(df['sigma_tail_rolling'])

# Create asset clusters
asset_codes = {a: i for i, a in enumerate(df['asset'].unique())}
df['asset_id'] = df['asset'].map(asset_codes)

print("=" * 72)
print("γ ROBUSTNESS: FULL-SAMPLE vs. ROLLING σ_tail")
print(f"N = {len(df)} cells")
print("=" * 72)

# Model 1: Full-sample σ_tail (v2 headline)
X1 = df[['log_nalpha', 'log_sigma_full']].copy()
X1 = sm.add_constant(X1)
mod1 = sm.OLS(df['log_sd'], X1).fit(cov_type='cluster',
              cov_kwds={'groups': df['asset_id']})

print("\n--- Model 1: Full-sample σ_tail (v2 headline) ---")
print(f"  b̂ (log(nα)):      {mod1.params['log_nalpha']:.4f} "
      f"(SE = {mod1.bse['log_nalpha']:.4f})")
print(f"  γ̂ (log(σ_tail)):  {mod1.params['log_sigma_full']:.4f} "
      f"(SE = {mod1.bse['log_sigma_full']:.4f})")
t_gamma1 = (mod1.params['log_sigma_full'] - 1) / mod1.bse['log_sigma_full']
print(f"  t(γ=1):            {t_gamma1:.4f}")
print(f"  R²:                {mod1.rsquared:.4f}")

# Model 2: Rolling σ_tail
X2 = df[['log_nalpha', 'log_sigma_roll']].copy()
X2 = sm.add_constant(X2)
mod2 = sm.OLS(df['log_sd'], X2).fit(cov_type='cluster',
              cov_kwds={'groups': df['asset_id']})

print("\n--- Model 2: Rolling σ_tail ---")
print(f"  b̂ (log(nα)):      {mod2.params['log_nalpha']:.4f} "
      f"(SE = {mod2.bse['log_nalpha']:.4f})")
print(f"  γ̂ (log(σ_roll)):  {mod2.params['log_sigma_roll']:.4f} "
      f"(SE = {mod2.bse['log_sigma_roll']:.4f})")
t_gamma2 = (mod2.params['log_sigma_roll'] - 1) / mod2.bse['log_sigma_roll']
print(f"  t(γ=1):            {t_gamma2:.4f}")
print(f"  R²:                {mod2.rsquared:.4f}")

# Model 3: With forecaster FEs + rolling σ_tail
df_fe = pd.get_dummies(df[['forecaster']], drop_first=True, dtype=float)
X3 = pd.concat([df[['log_nalpha', 'log_sigma_roll']], df_fe], axis=1)
X3 = sm.add_constant(X3)
mod3 = sm.OLS(df['log_sd'], X3).fit(cov_type='cluster',
              cov_kwds={'groups': df['asset_id']})

print("\n--- Model 3: Rolling σ_tail + Forecaster FE ---")
print(f"  b̂ (log(nα)):      {mod3.params['log_nalpha']:.4f} "
      f"(SE = {mod3.bse['log_nalpha']:.4f})")
print(f"  γ̂ (log(σ_roll)):  {mod3.params['log_sigma_roll']:.4f} "
      f"(SE = {mod3.bse['log_sigma_roll']:.4f})")
t_gamma3 = (mod3.params['log_sigma_roll'] - 1) / mod3.bse['log_sigma_roll']
print(f"  t(γ=1):            {t_gamma3:.4f}")
print(f"  R²:                {mod3.rsquared:.4f}")

# Summary comparison
print("\n\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  {'':30s} {'γ̂':>8s} {'SE':>8s} {'t(γ=1)':>8s} {'b̂':>8s}")
print(f"  {'Full-sample σ_tail':30s} {mod1.params['log_sigma_full']:>8.3f} "
      f"{mod1.bse['log_sigma_full']:>8.3f} {t_gamma1:>8.3f} "
      f"{mod1.params['log_nalpha']:>8.3f}")
print(f"  {'Rolling σ_tail':30s} {mod2.params['log_sigma_roll']:>8.3f} "
      f"{mod2.bse['log_sigma_roll']:>8.3f} {t_gamma2:>8.3f} "
      f"{mod2.params['log_nalpha']:>8.3f}")
print(f"  {'Rolling σ_tail + FE':30s} {mod3.params['log_sigma_roll']:>8.3f} "
      f"{mod3.bse['log_sigma_roll']:>8.3f} {t_gamma3:>8.3f} "
      f"{mod3.params['log_nalpha']:>8.3f}")

# Generate LaTeX table
print("\n\n--- LaTeX Table B.14 ---")
print(r"\begin{tabular}{l cccc}")
print(r"\toprule")
print(r"$\sigma_{\mathrm{tail}}$ measure & $\hat{\gamma}$ & SE & "
      r"$t(\gamma=1)$ & $\hat{b}$ \\")
print(r"\midrule")
print(f"Full-sample & {mod1.params['log_sigma_full']:.3f} & "
      f"{mod1.bse['log_sigma_full']:.3f} & {t_gamma1:.2f} & "
      f"{mod1.params['log_nalpha']:.3f} \\\\")
print(f"Rolling (250-day) & {mod2.params['log_sigma_roll']:.3f} & "
      f"{mod2.bse['log_sigma_roll']:.3f} & {t_gamma2:.2f} & "
      f"{mod2.params['log_nalpha']:.3f} \\\\")
print(f"Rolling + forecaster FE & {mod3.params['log_sigma_roll']:.3f} & "
      f"{mod3.bse['log_sigma_roll']:.3f} & {t_gamma3:.2f} & "
      f"{mod3.params['log_nalpha']:.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")

