"""
TimesFM first-stage extraction noise diagnostic.

Raw 9-decile quantile heads are not stored; we proxy quantile-fit quality
with mean(1/nu), the average inverse fitted Student-t degrees of freedom.
Low nu => long tail-extrapolation lever => more extraction noise.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as t_dist
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RECALIB = os.path.join(BASE, '..', 'data', 'recalib_results.csv')
TFM_DIR = "/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data/timesfm25"
OUT_DIR = os.path.join(BASE, '..', 'tables')


def main():
    recalib = pd.read_csv(RECALIB)
    tfm = recalib[
        (recalib['forecaster'] == 'TimesFM-2.5') & (recalib['alpha'] == 0.01)
    ][['asset', 'n_windows', 'detrended_sd', 'sigma_tail']].reset_index(drop=True)

    proxy_rows = []
    for _, row in tfm.iterrows():
        df = pd.read_parquet(os.path.join(TFM_DIR, f"{row['asset']}.parquet"))
        nu = df['df_student'].values
        valid = (nu > 2) & np.isfinite(nu) & (nu < 1e6)
        nu_v = nu[valid]
        proxy_rows.append({
            'asset': row['asset'],
            'mean_inv_nu': np.mean(1.0 / nu_v),
        })
    proxy = pd.DataFrame(proxy_rows)
    m = tfm.merge(proxy, on='asset')

    y = np.log(m['detrended_sd'])

    # Model A: baseline — log(sigma_tail) only
    XA = sm.add_constant(np.log(m['sigma_tail']))
    mA = sm.OLS(y, XA).fit(cov_type='HC1')

    # Model B: add log(mean(1/nu)) as first-stage noise proxy
    XB = pd.DataFrame({
        'const': 1.0,
        'log_sigma_tail': np.log(m['sigma_tail'].values),
        'log_inv_nu': np.log(m['mean_inv_nu'].values),
    })
    mB = sm.OLS(y, XB).fit(cov_type='HC1')

    print("Model A (baseline):")
    print(f"  gamma = {mA.params.iloc[1]:.3f} (SE {mA.bse.iloc[1]:.3f}), R² = {mA.rsquared:.3f}")
    print("Model B (+ first-stage proxy):")
    print(f"  gamma = {mB.params.iloc[1]:.3f} (SE {mB.bse.iloc[1]:.3f})")
    print(f"  c     = {mB.params.iloc[2]:.3f} (SE {mB.bse.iloc[2]:.3f}, t = {mB.tvalues.iloc[2]:.2f})")
    print(f"  R²    = {mB.rsquared:.3f}")

    # LaTeX table
    lines = [
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r" & (A) & (B) \\",
        r"\midrule",
        rf"$\log\sigma_{{\mathrm{{tail}}}}$ & {mA.params.iloc[1]:.3f} & {mB.params.iloc[1]:.3f} \\",
        rf" & ({mA.bse.iloc[1]:.3f}) & ({mB.bse.iloc[1]:.3f}) \\[3pt]",
        rf"$\log\overline{{\nu^{{-1}}}}$ & --- & {mB.params.iloc[2]:.3f} \\",
        rf" & & ({mB.bse.iloc[2]:.3f}) \\[3pt]",
        rf"$R^2$ & {mA.rsquared:.3f} & {mB.rsquared:.3f} \\",
        r"$N$ & 24 & 24 \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(lines) + "\n"
    outpath = os.path.join(OUT_DIR, "timesfm_mechanism.tex")
    with open(outpath, "w") as f:
        f.write(tex)
    print(f"\nTable written to {outpath}")


if __name__ == "__main__":
    main()
