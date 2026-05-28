"""
Task 4: Detrending sensitivity for fragile-comparison share.
The fragile comparison uses the plug-in bound sqrt(C1^2+C2^2)/sqrt(nalpha).
The bound itself (sigma_tail) is detrending-invariant, but the empirical
*dispersion* of r_hat under different detrending methods is used here as
an alternative noise scale to show sensitivity. The theoretical bound
remains the headline.

Deterministic seed: 42
Author: Daniel Traian Pele
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])
recalib = pd.read_csv(OUT / "data" / "recalib_results.csv")

WINDOW = 250
ALPHAS = [0.01, 0.025, 0.05]


def hp_filter(y, lam=1600):
    n = len(y)
    if n < 4:
        return y, np.zeros(n)
    I = np.eye(n)
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1; D[i, i+1] = -2; D[i, i+2] = 1
    trend = np.linalg.solve(I + lam * D.T @ D, y)
    return trend, y - trend


def detrend_series(r_hat, method):
    s = pd.Series(r_hat).reset_index(drop=True)
    if method == "MA-126":
        trend = s.rolling(6, center=True, min_periods=3).mean()
    elif method == "MA-252":
        trend = s.rolling(12, center=True, min_periods=6).mean()
    elif method == "MA-504":
        trend = s.rolling(24, center=True, min_periods=12).mean()
    elif method == "HP":
        t_arr, resid = hp_filter(s.values, lam=1600)
        return resid
    elif method == "Rolling median":
        trend = s.rolling(12, center=True, min_periods=6).median()
    elif method == "Raw":
        return s.values
    else:
        raise ValueError(method)
    resid = s - trend
    return resid.dropna().values


def compute_fragile_share_empirical(roll_df, recalib_df, alpha_val, method):
    """Fragile share using detrended SD as the noise scale.
    threshold = sqrt(dtr_sd_1^2 + dtr_sd_2^2)."""
    sub_roll = roll_df[roll_df["alpha"] == alpha_val]
    sub_recalib = recalib_df[recalib_df["alpha"] == alpha_val]
    assets = sub_recalib["asset"].unique()
    forecasters = sub_recalib["forecaster"].unique()

    cell_sd = {}
    cell_mean = {}
    for asset in assets:
        for fname in forecasters:
            r_series = sub_roll[(sub_roll["asset"] == asset) &
                                (sub_roll["forecaster"] == fname)]["r_hat"].values
            if len(r_series) < 10:
                continue
            resid = detrend_series(r_series, method)
            if len(resid) < 5:
                continue
            cell_sd[(asset, fname)] = np.std(resid, ddof=1)
            cell_mean[(asset, fname)] = np.mean(r_series)

    n_comp = 0
    n_frag = 0
    for asset in assets:
        for i, f1 in enumerate(forecasters):
            for f2 in forecasters[i+1:]:
                key1, key2 = (asset, f1), (asset, f2)
                if key1 not in cell_sd or key2 not in cell_sd:
                    continue
                diff = abs(cell_mean[key1] - cell_mean[key2])
                threshold = np.sqrt(cell_sd[key1]**2 + cell_sd[key2]**2)
                n_comp += 1
                if diff < threshold:
                    n_frag += 1

    pct = 100 * n_frag / n_comp if n_comp > 0 else np.nan
    return n_comp, n_frag, pct


def compute_fragile_share_theoretical(recalib_df, alpha_val):
    """Original fragile share using theoretical bound (detrending-invariant)."""
    sub = recalib_df[recalib_df["alpha"] == alpha_val]
    assets = sub["asset"].unique()
    forecasters = sub["forecaster"].unique()
    n_comp = 0
    n_frag = 0
    for asset in assets:
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
    return n_comp, n_frag, 100 * n_frag / n_comp if n_comp > 0 else np.nan


def paired_block_bootstrap(roll_df, recalib_df, alpha_val, block_len=21,
                           n_boot=1000):
    sub_roll = roll_df[roll_df["alpha"] == alpha_val]
    sub_recalib = recalib_df[recalib_df["alpha"] == alpha_val]
    assets = sub_recalib["asset"].unique()
    forecasters = sorted(sub_recalib["forecaster"].unique())

    boot_fragile = np.zeros(n_boot)

    for b in range(n_boot):
        n_comp = 0
        n_frag = 0
        for asset in assets:
            rhat_by_fcast = {}
            for fname in forecasters:
                s = sub_roll[(sub_roll["asset"] == asset) &
                             (sub_roll["forecaster"] == fname)]
                s = s.sort_values("date")
                rhat_by_fcast[fname] = s["r_hat"].values

            if not rhat_by_fcast:
                continue
            min_len = min(len(v) for v in rhat_by_fcast.values())
            if min_len < 10:
                continue

            for fname in forecasters:
                rhat_by_fcast[fname] = rhat_by_fcast[fname][:min_len]

            # Block bootstrap indices (joint across forecasters)
            n_blocks = max(1, min_len // block_len)
            block_starts = np.random.randint(0, max(1, min_len - block_len + 1),
                                              size=n_blocks)
            boot_idx = np.concatenate([np.arange(s, min(s + block_len, min_len))
                                        for s in block_starts])[:min_len]

            boot_means = {fname: rhat_by_fcast[fname][boot_idx].mean()
                         for fname in forecasters}

            for i, f1 in enumerate(forecasters):
                for f2 in forecasters[i+1:]:
                    rc1 = sub_recalib[(sub_recalib["asset"] == asset) &
                                      (sub_recalib["forecaster"] == f1)]
                    rc2 = sub_recalib[(sub_recalib["asset"] == asset) &
                                      (sub_recalib["forecaster"] == f2)]
                    if len(rc1) == 0 or len(rc2) == 0:
                        continue
                    diff = abs(boot_means[f1] - boot_means[f2])
                    c1 = rc1.iloc[0]["bound"]
                    c2 = rc2.iloc[0]["bound"]
                    threshold = np.sqrt(c1**2 + c2**2)
                    n_comp += 1
                    if diff < threshold:
                        n_frag += 1

        boot_fragile[b] = 100 * n_frag / n_comp if n_comp > 0 else np.nan

    return np.nanmean(boot_fragile), np.nanstd(boot_fragile)


# --- Main computation ---
METHODS = ["MA-126", "MA-252", "MA-504", "HP", "Rolling median", "Raw"]

print("="*70)
print("Table 7: Fragile-comparison share by detrending method")
print("(using detrended SD as noise scale)")
print("="*70)

table_rows = []
for method in METHODS:
    row = {"Detrending": method}
    for alpha in ALPHAS:
        nc, nf, pct = compute_fragile_share_empirical(roll, recalib, alpha, method)
        row[f"alpha_{alpha}"] = pct
        print(f"  {method:>16s}, alpha={alpha}: {nf}/{nc} ({pct:.1f}%)")
    table_rows.append(row)

tab_df = pd.DataFrame(table_rows)

# Also compute headline (theoretical bound)
print("\nHeadline (theoretical bound):")
for alpha in ALPHAS:
    nc, nf, pct = compute_fragile_share_theoretical(recalib, alpha)
    print(f"  alpha={alpha}: {nf}/{nc} ({pct:.1f}%)")

# Bootstrap
print(f"\nComputing paired block bootstrap (1000 reps, block=21)...")
boot_mean, boot_sd = paired_block_bootstrap(roll, recalib, 0.025,
                                             block_len=21, n_boot=1000)
print(f"  Bootstrap fragile share at alpha=2.5%: {boot_mean:.1f}% (SD = {boot_sd:.1f}%)")

# --- Generate LaTeX ---
method_labels = {
    "MA-126": "MA-126",
    "MA-252": "MA-252 (headline)",
    "MA-504": "MA-504",
    "HP": "HP filter ($\\lambda{=}1600$)",
    "Rolling median": "Rolling median 252",
    "Raw": "Raw (no detrending)",
}

tex_lines = [
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Detrending & Fragile @ $\alpha{=}1\%$ & "
    r"Fragile @ $\alpha{=}2.5\%$ & Fragile @ $\alpha{=}5\%$ \\",
    r"\midrule",
]

for _, row in tab_df.iterrows():
    label = method_labels.get(row["Detrending"], row["Detrending"])
    tex_lines.append(
        f"  {label} & {row['alpha_0.01']:.1f}\\% & "
        f"{row['alpha_0.025']:.1f}\\% & {row['alpha_0.05']:.1f}\\% \\\\"
    )

tex_lines.append(r"\midrule")
tex_lines.append(
    f"  Paired bootstrap (block${{=}}$21) & --- & "
    f"{boot_mean:.1f}\\% & --- \\\\"
)
tex_lines += [r"\bottomrule", r"\end{tabular}"]
tex = "\n".join(tex_lines)
(OUT / "tables" / "detrending_fragile_robustness.tex").write_text(tex)
print(f"\nTable saved to tables/detrending_fragile_robustness.tex")

vals_025 = [r["alpha_0.025"] for _, r in tab_df.iterrows()]
lo, hi = min(vals_025), max(vals_025)
headline = tab_df[tab_df["Detrending"] == "MA-252"]["alpha_0.025"].values[0]
print(f"\nFragile share at alpha=2.5% range: [{lo:.1f}%, {hi:.1f}%]")
print(f"Headline = {headline:.1f}%")
print(f"Bootstrap = {boot_mean:.1f}%")
