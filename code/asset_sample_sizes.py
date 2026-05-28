"""Addition 2: Asset-specific sample-size table.

Uses empirical sigma_tail from GJR-GARCH-t at alpha=1%.
Tolerance epsilon in return units.  Required n from n >= sigma_tail^2/(alpha*epsilon^2).

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
df = pd.read_csv(OUT / "data" / "recalib_results.csv")

alpha = 0.01
sub = df[(df["forecaster"] == "GJR-GARCH-t") & (df["alpha"] == alpha)].copy()
sub = sub.sort_values("sigma_tail")

eps_levels = [0.0025, 0.005, 0.01, 0.02]
eps_labels = ["25\\,bp", "50\\,bp", "100\\,bp", "200\\,bp"]

rows = []
for _, r in sub.iterrows():
    asset = r["asset"]
    sig = r["sigma_tail"]
    bound_250 = sig / np.sqrt(250 * alpha)
    row = {"asset": asset, "sigma_tail": sig, "bound_250": bound_250}
    for eps in eps_levels:
        n_req = int(np.ceil(sig ** 2 / (alpha * eps ** 2)))
        row[f"n_{eps}"] = n_req
    rows.append(row)

adf = pd.DataFrame(rows)
adf.to_csv(OUT / "tables" / "asset_sample_sizes.csv", index=False)

def fmt_n(n):
    if n >= 1000:
        return f"{n:,}".replace(",", "{,}")
    return str(n)

lines = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    (r"Asset & $\hat{\sigma}_{\mathrm{tail}}$ (\%) "
     r"& $\varepsilon_{250}$ (bp) "
     + "".join(f"& $\\varepsilon{{=}}{l}$ " for l in eps_labels) + r"\\"),
    r"\midrule",
]
for _, r in adf.iterrows():
    sig_pct = r["sigma_tail"] * 100
    bound_bp = r["bound_250"] * 10000
    cols = f"  {r['asset']} & {sig_pct:.2f} & {bound_bp:.0f}"
    for eps in eps_levels:
        cols += f" & {fmt_n(r[f'n_{eps}'])}"
    cols += r" \\"
    lines.append(cols)

med_sig = np.median(adf["sigma_tail"].values)
med_bound = np.median(adf["bound_250"].values)
lines.append(r"\midrule")
med_cols = f"  \\textit{{Median}} & {med_sig*100:.2f} & {med_bound*10000:.0f}"
for eps in eps_levels:
    n_med = int(np.ceil(med_sig ** 2 / (alpha * eps ** 2)))
    med_cols += f" & {fmt_n(n_med)}"
med_cols += r" \\"
lines.append(med_cols)
lines += [r"\bottomrule", r"\end{tabular}"]

(OUT / "tables" / "asset_sample_sizes.tex").write_text("\n".join(lines))

print(f"Assets: {len(adf)}")
print(f"Median sigma_tail: {med_sig:.4f} ({med_sig*100:.2f}%)")
print(f"Median bound at n=250: {med_bound:.4f} ({med_bound*10000:.0f} bp)")
print(f"sigma_tail range: {adf['sigma_tail'].min()*100:.2f}% to {adf['sigma_tail'].max()*100:.2f}%")
for eps, lab in zip(eps_levels, ["25bp", "50bp", "100bp", "200bp"]):
    n_min = adf[f"n_{eps}"].min()
    n_max = adf[f"n_{eps}"].max()
    n_med = int(np.ceil(med_sig ** 2 / (alpha * eps ** 2)))
    print(f"  n for {lab}: [{n_min}, {n_max}], median={n_med}")

btc = adf[adf["asset"] == "BTC"].iloc[0]
sp = adf[adf["asset"] == "SP500"].iloc[0]
print(f"\nSP500: sigma_tail={sp['sigma_tail']*100:.2f}%, bound@250={sp['bound_250']*10000:.0f}bp, n@50bp={sp['n_0.005']}")
print(f"BTC:   sigma_tail={btc['sigma_tail']*100:.2f}%, bound@250={btc['bound_250']*10000:.0f}bp, n@50bp={btc['n_0.005']}")
print("Table saved.")
