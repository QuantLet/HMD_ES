"""Regenerate Addition 1 figures and table with correct CI stats.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
df = pd.read_csv(OUT / "tables" / "window_scaling.csv")

# ── Recompute slopes ──
slopes = []
for asset in sorted(df["asset"].unique()):
    sub = df[(df["asset"] == asset) & (~df["dropped"])]
    if len(sub) < 3:
        continue
    x = np.log(sub["n"].values.astype(float))
    y = np.log(sub["raw_sd"].values)
    res = linregress(x, y)
    lo, hi = res.slope - 1.96 * res.stderr, res.slope + 1.96 * res.stderr
    slopes.append({
        "asset": asset, "slope": res.slope, "ci_lo": lo, "ci_hi": hi,
        "r2": res.rvalue ** 2, "contains": lo <= -0.5 <= hi,
        "intercept": res.intercept, "se": res.stderr,
    })

sdf = pd.DataFrame(slopes)
sl = sdf["slope"].values
med = np.median(sl)
q1, q3 = np.percentile(sl, [25, 75])
n_contains = int(sdf["contains"].sum())
pct_contains = n_contains / len(sdf) * 100

print(f"Median slope: {med:.3f}")
print(f"IQR: [{q1:.3f}, {q3:.3f}]")
print(f"-0.5 in CI: {n_contains}/{len(sdf)} = {pct_contains:.0f}%")

# ── Figure 5a: Histogram ──
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(sl, bins=12, range=(-1.5, 0.0), color="#003DA5", alpha=0.7,
        edgecolor="white", linewidth=0.8)
ax.axvline(-0.5, color="black", linewidth=2, label=r"Theoretical $-1/2$")
ax.axvline(med, color="#C8102E", linewidth=2, linestyle="--",
           label=f"Median $= {med:.2f}$")

txt = (f"Median: {med:.2f}\n"
       f"IQR: [{q1:.2f}, {q3:.2f}]\n"
       f"$-0.5$ in CI: {pct_contains:.0f}% ({n_contains}/23)")
ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

ax.set_xlabel(r"OLS slope $\hat{b}$", fontsize=12)
ax.set_ylabel("Number of assets", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figures" / "HMD_SlopeHistogram.pdf",
            bbox_inches="tight", transparent=True)
fig.savefig(OUT / "figures" / "HMD_SlopeHistogram.png",
            bbox_inches="tight", transparent=True, dpi=150)
plt.close()
print("Histogram saved.")

# ── Figure 5b: 3-panel illustration ──
panel_assets = ["SP500", "BTC", "NATGAS"]
colors = {"SP500": "#003DA5", "BTC": "#C8102E", "NATGAS": "#DC143C"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
for ax, asset in zip(axes, panel_assets):
    sub = df[(df["asset"] == asset) & (~df["dropped"])]
    sl_row = sdf[sdf["asset"] == asset].iloc[0]
    c = colors[asset]

    x = np.log(sub["n"].values.astype(float))
    y = np.log(sub["raw_sd"].values)

    ax.scatter(x, y, color=c, s=50, zorder=3)

    # OLS fit
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = sl_row["intercept"] + sl_row["slope"] * x_fit
    ax.plot(x_fit, y_fit, color=c, linewidth=1.5, zorder=2,
            label=r"$\hat{b} = " + f"{sl_row['slope']:.2f}$")

    # -0.5 reference
    y_ref = y.mean() - 0.5 * (x_fit - x.mean())
    ax.plot(x_fit, y_ref, color="gray", linewidth=1.2, linestyle="--",
            label="$b = -0.50$", zorder=1)

    ax.set_xlabel(r"$\log(n)$", fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel(r"$\log(\mathrm{SD})$", fontsize=11)
    ax.set_title(asset, fontsize=12)
    ci_str = f"[{sl_row['ci_lo']:.2f}, {sl_row['ci_hi']:.2f}]"
    ax.text(0.03, 0.03, f"95% CI: {ci_str}", transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom")
    ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig(OUT / "figures" / "HMD_WindowScaling.pdf",
            bbox_inches="tight", transparent=True)
fig.savefig(OUT / "figures" / "HMD_WindowScaling.png",
            bbox_inches="tight", transparent=True, dpi=150)
plt.close()
print("Panel figure saved.")

# ── Table ──
sdf_sorted = sdf.sort_values("slope")
lines = [
    r"\begin{tabular}{lrrrrrrrc}",
    r"\toprule",
    (r"Asset & $n{=}250$ & $n{=}500$ & $n{=}1000$ & $n{=}2000$ "
     r"& $\hat{b}$ & 95\% CI & $R^2$ & $-0.5 \in$ CI \\"),
    r"\midrule",
]
for _, sl_row in sdf_sorted.iterrows():
    asset = sl_row["asset"]
    sub = df[(df["asset"] == asset) & (~df["dropped"])]
    vals = {}
    for n in [250, 500, 1000, 2000]:
        row = sub[sub["n"] == n]
        if len(row) > 0 and not np.isnan(row.iloc[0]["raw_sd"]):
            vals[n] = f"{row.iloc[0]['raw_sd']:.4f}"
        else:
            vals[n] = "--"
    ci = f"[{sl_row['ci_lo']:.2f}, {sl_row['ci_hi']:.2f}]"
    yn = "Y" if sl_row["contains"] else "N"
    lines.append(
        f"  {asset} & {vals[250]} & {vals[500]} & "
        f"{vals[1000]} & {vals[2000]} & "
        f"${sl_row['slope']:.2f}$ & ${ci}$ & "
        f"${sl_row['r2']:.3f}$ & {yn} \\\\"
    )
lines.append(r"\midrule")
lines.append(
    f"  \\textit{{Median}} & & & & & ${med:.2f}$ & "
    f"IQR $[{q1:.2f},{q3:.2f}]$ & & {pct_contains:.0f}\\% \\\\"
)
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables" / "window_scaling_24assets.tex").write_text("\n".join(lines))
print("Table saved.")
