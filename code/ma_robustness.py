"""
Recompute detrended SD and RMSE-to-bound ratio using MA-126 and MA-504
detrending windows (6 and 24 steps at step=21 days).
Baseline is MA-252 (12 steps).

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])
res  = pd.read_csv(OUT / "data" / "recalib_results.csv")

MA_CONFIGS = {
    "MA-126": 6,
    "MA-252": 12,
    "MA-504": 24,
}

WINDOW = 250

rows = []
for (asset, forecaster, alpha), grp in roll.groupby(["asset", "forecaster", "alpha"]):
    grp = grp.sort_values("date")
    s = grp["r_hat"].values

    base = res[(res["asset"] == asset) &
               (res["forecaster"] == forecaster) &
               (res["alpha"] == alpha)]
    if len(base) == 0:
        continue
    bound = base.iloc[0]["bound"]
    if np.isnan(bound) or bound == 0:
        continue

    for ma_name, smooth_w in MA_CONFIGS.items():
        ser = pd.Series(s)
        if len(ser) >= 2 * smooth_w:
            trend = ser.rolling(smooth_w, center=True,
                                min_periods=smooth_w // 2).mean()
            resid = ser - trend
            dsd = resid.dropna().std(ddof=1)
        else:
            dsd = ser.std(ddof=1)

        ratio = dsd / bound
        rows.append({
            "asset": asset,
            "forecaster": forecaster,
            "alpha": alpha,
            "ma_window": ma_name,
            "detrended_sd": dsd,
            "bound": bound,
            "ratio": ratio,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "tables" / "ma_robustness.csv", index=False)

# Table B.1: median ratio at alpha=1% by forecaster x MA window
sub1 = df[df["alpha"] == 0.01]
pivot = sub1.pivot_table(index="forecaster", columns="ma_window",
                         values="ratio", aggfunc="median")
pivot = pivot[["MA-126", "MA-252", "MA-504"]]
f_order = ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]
pivot = pivot.reindex([f for f in f_order if f in pivot.index])

print("Table B.1: Median RMSE-to-bound ratio at alpha=1%")
print(pivot.round(2).to_string())
print()

# Max absolute deviation from baseline
for f in pivot.index:
    base_val = pivot.loc[f, "MA-252"]
    for col in ["MA-126", "MA-504"]:
        dev = abs(pivot.loc[f, col] - base_val)
        print(f"  {f} | {col} vs MA-252: dev = {dev:.3f}")

deviations = []
for f in pivot.index:
    base_val = pivot.loc[f, "MA-252"]
    for col in ["MA-126", "MA-504"]:
        deviations.append(abs(pivot.loc[f, col] - base_val))
delta_max = max(deviations)
print(f"\nDelta_max = {delta_max:.2f}")

# Check ranking preservation
print("\nRanking by median ratio (ascending) at alpha=1%:")
for ma in ["MA-126", "MA-252", "MA-504"]:
    ranking = pivot[ma].sort_values()
    print(f"  {ma}: {' < '.join(ranking.index)}")

# Generate LaTeX table
tex_lines = [
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Forecaster & MA-126 & MA-252 & MA-504 \\",
    r"\midrule",
]
for f in pivot.index:
    vals = " & ".join(f"{pivot.loc[f, c]:.2f}" for c in ["MA-126", "MA-252", "MA-504"])
    tex_lines.append(f"  {f} & {vals} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")

tex_path = OUT / "tables" / "ma_robustness.tex"
tex_path.write_text("\n".join(tex_lines) + "\n")
print(f"\nLaTeX table written to {tex_path}")
