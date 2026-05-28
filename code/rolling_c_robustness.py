"""
Task 4.2: Rolling Ĉ robustness.
Re-estimate Ĉ = σ_tail on each calibration window and recompute R.
Compare to the full-sample Ĉ headline results.
"""
import pandas as pd
import numpy as np

# Load data
recalib = pd.read_csv('/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/data/recalib_results.csv')
rolling = pd.read_csv('/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData/data/rolling_estimates.csv')

# Filter to rows with rolling sigma_tail
roll = rolling[rolling['sigma_tail_w'].notna()].copy()

results = []
for (asset, fcast, alpha), grp in roll.groupby(['asset', 'forecaster', 'alpha']):
    if len(grp) < 10:
        continue
    
    n_alpha = 250 * alpha
    
    # Detrend r_hat with 252-period MA
    r = grp['r_hat'].values
    if len(r) < 12:
        continue
    ma = pd.Series(r).rolling(12, min_periods=6, center=True).mean().values
    detrended = r - ma
    detrended = detrended[~np.isnan(detrended)]
    if len(detrended) < 5:
        continue
    
    detrended_sd = np.std(detrended, ddof=1)
    
    # Full-sample bound (from recalib file)
    row = recalib[(recalib['asset'] == asset) & 
                   (recalib['forecaster'] == fcast) & 
                   (np.isclose(recalib['alpha'], alpha))]
    if len(row) == 0:
        continue
    sigma_tail_full = row['sigma_tail'].values[0]
    bound_full = sigma_tail_full / np.sqrt(n_alpha)
    R_full = detrended_sd / bound_full if bound_full > 0 else np.nan
    
    # Rolling bound: median of rolling sigma_tail_w / √(nα)
    sigma_tail_roll = grp['sigma_tail_w'].median()
    bound_roll = sigma_tail_roll / np.sqrt(n_alpha)
    R_roll = detrended_sd / bound_roll if bound_roll > 0 else np.nan
    
    results.append({
        'asset': asset, 'forecaster': fcast, 'alpha': alpha,
        'R_full': R_full, 'R_roll': R_roll,
        'sigma_full': sigma_tail_full, 'sigma_roll': sigma_tail_roll
    })

df = pd.DataFrame(results)

# Summary by forecaster
print("=" * 70)
print("ROLLING Ĉ ROBUSTNESS")
print("=" * 70)

for alpha in [0.01, 0.025, 0.05]:
    sub = df[np.isclose(df['alpha'], alpha)]
    print(f"\n--- α = {alpha} ---")
    for fcast in ['GJR-GARCH-t', 'TimesFM-2.5', 'Chronos-Small', 'Moirai-2.0']:
        fsub = sub[sub['forecaster'] == fcast]
        if len(fsub) == 0:
            continue
        med_full = fsub['R_full'].median()
        med_roll = fsub['R_roll'].median()
        print(f"  {fcast:20s}: R_full={med_full:.3f}  R_roll={med_roll:.3f}  "
              f"diff={med_roll-med_full:+.3f}  n={len(fsub)}")

# Fragile share comparison at α=2.5%
print("\n\n--- Fragile share at α=2.5% ---")
sub = df[np.isclose(df['alpha'], 0.025)]
forecasters = sub['forecaster'].unique()
assets = sub['asset'].unique()

n_fragile_full = 0
n_fragile_roll = 0
n_total = 0
for i, f1 in enumerate(forecasters):
    for j, f2 in enumerate(forecasters):
        if j <= i:
            continue
        for asset in assets:
            r1 = sub[(sub['forecaster'] == f1) & (sub['asset'] == asset)]
            r2 = sub[(sub['forecaster'] == f2) & (sub['asset'] == asset)]
            if len(r1) == 0 or len(r2) == 0:
                continue
            # Check fragility using full-sample Ĉ
            row1 = recalib[(recalib['asset'] == asset) & 
                           (recalib['forecaster'] == f1) &
                           (np.isclose(recalib['alpha'], 0.025))]
            row2 = recalib[(recalib['asset'] == asset) & 
                           (recalib['forecaster'] == f2) &
                           (np.isclose(recalib['alpha'], 0.025))]
            if len(row1) == 0 or len(row2) == 0:
                continue
            
            c1 = row1['sigma_tail'].values[0]
            c2 = row2['sigma_tail'].values[0]
            n_alpha = 250 * 0.025
            bound = np.sqrt(c1**2 + c2**2) / np.sqrt(n_alpha)
            diff = abs(row1['r_hat_mean'].values[0] - row2['r_hat_mean'].values[0])
            fragile_full = diff < bound
            
            # Using rolling Ĉ
            c1r = r1['sigma_roll'].values[0] if 'sigma_roll' in r1.columns else c1
            c2r = r2['sigma_roll'].values[0] if 'sigma_roll' in r2.columns else c2
            bound_r = np.sqrt(c1r**2 + c2r**2) / np.sqrt(n_alpha)
            fragile_roll = diff < bound_r
            
            n_total += 1
            n_fragile_full += fragile_full
            n_fragile_roll += fragile_roll

print(f"  Full-sample Ĉ:  {n_fragile_full}/{n_total} = "
      f"{100*n_fragile_full/n_total:.1f}%")
print(f"  Rolling Ĉ:      {n_fragile_roll}/{n_total} = "
      f"{100*n_fragile_roll/n_total:.1f}%")

# LaTeX summary table
print("\n\n--- LaTeX table ---")
print(r"\begin{tabular}{l rr rr}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{$\alpha = 1\%$} & \multicolumn{2}{c}{$\alpha = 2.5\%$} \\")
print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
print(r"Forecaster & $R_{\mathrm{full}}$ & $R_{\mathrm{roll}}$ "
      r"& $R_{\mathrm{full}}$ & $R_{\mathrm{roll}}$ \\")
print(r"\midrule")
for fcast in ['GJR-GARCH-t', 'TimesFM-2.5', 'Chronos-Small', 'Moirai-2.0']:
    vals = []
    for alpha in [0.01, 0.025]:
        fsub = df[(df['forecaster'] == fcast) & np.isclose(df['alpha'], alpha)]
        if len(fsub) > 0:
            vals.extend([fsub['R_full'].median(), fsub['R_roll'].median()])
        else:
            vals.extend([float('nan'), float('nan')])
    print(f"{fcast} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {vals[3]:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")

