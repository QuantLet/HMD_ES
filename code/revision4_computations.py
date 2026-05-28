"""
Revision 4 computations: fill all TBD placeholders in paper.tex.

1. Cutoff-multiplier sensitivity (kappa = 0.5, 1.0, 1.5, 2.0)
2. Results excluding Chronos-Small
3. VaR-miscalibration simulation (GARCH-t_5, constant vs conditional VaR)
4. Detrending illustration figure (S&P 500)
5. Forest plot of asset-specific sample sizes

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import t as tdist

OUT = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")
TABDIR = OUT / "tables"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(exist_ok=True)

MAINBLUE = "#003DA5"
IDARED   = "#C8102E"
FOREST   = "#228B22"
CRIMSON  = "#DC143C"

SAVE_KW = dict(bbox_inches="tight", transparent=True)
plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
})

df = pd.read_csv(OUT / "data" / "recalib_results.csv")
roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv", parse_dates=["date"])

print("=" * 70)
print("REVISION 4 COMPUTATIONS")
print("=" * 70)


# =====================================================================
# 1. CUTOFF-MULTIPLIER SENSITIVITY
# =====================================================================
print("\n1. CUTOFF-MULTIPLIER SENSITIVITY")
print("-" * 40)

kappa_results = []
for kappa in [0.5, 1.0, 1.5, 2.0]:
    for alpha_val in [0.01, 0.025, 0.05]:
        sub = df[df["alpha"] == alpha_val]
        forecasters = sub["forecaster"].unique()
        assets_list = sub["asset"].unique()
        n_comp = 0
        n_frag = 0
        for asset in assets_list:
            ad = sub[sub["asset"] == asset]
            for i, f1 in enumerate(forecasters):
                for f2 in forecasters[i + 1:]:
                    row1 = ad[ad["forecaster"] == f1]
                    row2 = ad[ad["forecaster"] == f2]
                    if len(row1) == 0 or len(row2) == 0:
                        continue
                    diff = abs(row1.iloc[0]["r_hat_mean"] - row2.iloc[0]["r_hat_mean"])
                    c1 = row1.iloc[0]["bound"]
                    c2 = row2.iloc[0]["bound"]
                    threshold = kappa * np.sqrt(c1**2 + c2**2)
                    n_comp += 1
                    if diff < threshold:
                        n_frag += 1
        pct = 100 * n_frag / n_comp if n_comp > 0 else 0
        kappa_results.append((kappa, alpha_val, n_comp, n_frag, pct))
        if alpha_val == 0.025:
            print(f"  kappa={kappa:.1f}, alpha=2.5%: {n_frag}/{n_comp} ({pct:.1f}%)")

kappa_df = pd.DataFrame(kappa_results,
                        columns=["kappa", "alpha", "n_comp", "n_frag", "pct"])

k025 = kappa_df[kappa_df["alpha"] == 0.025]
tex = r"""\begin{tabular}{cccc}
\toprule
$\kappa$ & Precision-fragile & Share (\%) & $n$ comparisons \\
\midrule"""
for _, row in k025.iterrows():
    tex += f"\n  {row['kappa']:.1f} & {int(row['n_frag'])} & {row['pct']:.1f}\\% & {int(row['n_comp'])} \\\\"
tex += r"""
\bottomrule
\end{tabular}"""
(TABDIR / "kappa_sensitivity.tex").write_text(tex)
print("  kappa_sensitivity.tex saved")


# =====================================================================
# 2. RESULTS EXCLUDING CHRONOS-SMALL
# =====================================================================
print("\n2. RESULTS EXCLUDING CHRONOS-SMALL")
print("-" * 40)

def compute_fragile_share(data, alpha_val, kappa=1.0):
    sub = data[data["alpha"] == alpha_val]
    forecasters = sub["forecaster"].unique()
    assets_list = sub["asset"].unique()
    n_comp = 0
    n_frag = 0
    for asset in assets_list:
        ad = sub[sub["asset"] == asset]
        for i, f1 in enumerate(forecasters):
            for f2 in forecasters[i + 1:]:
                row1 = ad[ad["forecaster"] == f1]
                row2 = ad[ad["forecaster"] == f2]
                if len(row1) == 0 or len(row2) == 0:
                    continue
                diff = abs(row1.iloc[0]["r_hat_mean"] - row2.iloc[0]["r_hat_mean"])
                c1 = row1.iloc[0]["bound"]
                c2 = row2.iloc[0]["bound"]
                threshold = kappa * np.sqrt(c1**2 + c2**2)
                n_comp += 1
                if diff < threshold:
                    n_frag += 1
    pct = 100 * n_frag / n_comp if n_comp > 0 else 0
    return n_comp, n_frag, pct

all_nc, all_nf, all_pct = compute_fragile_share(df, 0.025)
df_no_chronos = df[df["forecaster"] != "Chronos-Small"]
excl_nc, excl_nf, excl_pct = compute_fragile_share(df_no_chronos, 0.025)
df_calib = df[df["forecaster"].isin(["GJR-GARCH-t", "Moirai-2.0"])]
calib_nc, calib_nf, calib_pct = compute_fragile_share(df_calib, 0.025)

all_R_median = df[df["alpha"] == 0.025]["ratio_detrended"].median()
excl_R_median = df_no_chronos[df_no_chronos["alpha"] == 0.025]["ratio_detrended"].median()
calib_R_median = df_calib[df_calib["alpha"] == 0.025]["ratio_detrended"].median()

cc = pd.read_csv(OUT / "tables" / "christoffersen_diagnostic.csv")
cc_clean = cc.dropna(subset=["cc_stat", "ratio_R"])
cc_clean = cc_clean[cc_clean["cc_stat"] > 0]
rho_all, p_all = stats.spearmanr(np.log(cc_clean["cc_stat"]), cc_clean["ratio_R"])
cc_no_chr = cc_clean[cc_clean["forecaster"] != "Chronos-Small"]
rho_excl, p_excl = stats.spearmanr(np.log(cc_no_chr["cc_stat"]), cc_no_chr["ratio_R"])
cc_calib = cc_clean[cc_clean["forecaster"].isin(["GJR-GARCH-t", "Moirai-2.0"])]
rho_calib, p_calib = (stats.spearmanr(np.log(cc_calib["cc_stat"]), cc_calib["ratio_R"])
                       if len(cc_calib) >= 3 else (np.nan, np.nan))

print(f"  All:       fragile={all_pct:.1f}% ({all_nf}/{all_nc}), R={all_R_median:.2f}, rho={rho_all:.3f}")
print(f"  Excl Chr:  fragile={excl_pct:.1f}% ({excl_nf}/{excl_nc}), R={excl_R_median:.2f}, rho={rho_excl:.3f}")
print(f"  Calib:     fragile={calib_pct:.1f}% ({calib_nf}/{calib_nc}), R={calib_R_median:.2f}, rho={rho_calib:.3f}")

def fmt_rho(rho, p):
    if np.isnan(rho):
        return "---"
    sig = "^{***}" if p < 0.001 else ("^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else ""))
    return f"${rho:.3f}{sig}$"

tex_excl = r"""\begin{tabular}{lccc}
\toprule
Sample & Precision-fragile share & Median $R$ & VaR-diagnostic $\rho$ \\
\midrule"""
tex_excl += f"\n  All forecasters & {all_pct:.1f}\\% & {all_R_median:.2f} & {fmt_rho(rho_all, p_all)} \\\\"
tex_excl += f"\n  Excl.\\ Chronos-Small & {excl_pct:.1f}\\% & {excl_R_median:.2f} & {fmt_rho(rho_excl, p_excl)} \\\\"
tex_excl += f"\n  Only calibrated & {calib_pct:.1f}\\% & {calib_R_median:.2f} & {fmt_rho(rho_calib, p_calib)} \\\\"
tex_excl += r"""
\bottomrule
\end{tabular}"""
(TABDIR / "excl_chronos.tex").write_text(tex_excl)
print("  excl_chronos.tex saved")


# =====================================================================
# 3. VAR-MISCALIBRATION SIMULATION
# =====================================================================
print("\n3. VAR-MISCALIBRATION SIMULATION")
print("-" * 40)

# Design: GARCH(1,1)-t_5. The model produces VaR/ES forecasts using a
# blend of conditional and unconditional sigma:
#   sigma_model = (1-delta)*sigma_t + delta*sigma_uncond
# delta=0: perfect conditional model (oracle)
# delta=1: unconditional model (Historical Simulation analog)
#
# With delta>0, the model's VaR/ES don't track volatility regimes.
# The FZ correction r_hat = mean(X|X≤VaR_model) - ES_model absorbs
# this misalignment, creating excess dispersion in the correction path.
#
# We measure the SD of r_hat across rolling windows and compare to the
# oracle-implied benchmark sigma_tail/sqrt(n*alpha).

np.random.seed(42)
alpha_sim = 0.025
nu = 5
omega = 1e-6
alpha_g = 0.10
beta_g = 0.85
n_window = 250
step = 21
q_alpha = tdist.ppf(alpha_sim, nu)
pdf_at_q = tdist.pdf(q_alpha, nu)
es_factor = -pdf_at_q / alpha_sim * (nu + q_alpha**2) / (nu - 1)

n_sims = 30
T_total = 12000
burn = 2000

results_sim = []
for delta_label, delta_val in [("Correct", 0.0),
                                ("Mild", 0.70),
                                ("Moderate", 0.85),
                                ("Severe", 1.00)]:
    all_R_raw = []
    all_hit = []
    all_uc = []

    for sim_i in range(n_sims):
        rng = np.random.default_rng(42 + sim_i)

        sigma2 = np.zeros(T_total)
        sigma_uncond2 = omega / (1 - alpha_g - beta_g)
        sigma2[0] = sigma_uncond2
        eps = tdist.rvs(nu, size=T_total, random_state=rng)
        r = np.zeros(T_total)
        for t in range(1, T_total):
            sigma2[t] = omega + alpha_g * r[t - 1]**2 + beta_g * sigma2[t - 1]
            r[t] = np.sqrt(sigma2[t]) * eps[t]

        r = r[burn:]
        sigma2 = sigma2[burn:]
        sigma = np.sqrt(sigma2)
        sigma_uncond = np.sqrt(sigma_uncond2)
        T = len(r)

        # Model sigma: blend of conditional and unconditional
        sigma_model = (1 - delta_val) * sigma + delta_val * sigma_uncond
        var_model = sigma_model * q_alpha
        es_model = sigma_model * es_factor

        # True VaR for sigma_tail computation
        var_true = sigma * q_alpha

        # Rolling windows
        starts = list(range(0, T - n_window + 1, step))
        corrections = []
        hit_rates = []

        for s in starts:
            wr = r[s:s + n_window]
            wv = var_model[s:s + n_window]
            we = es_model[s:s + n_window]

            hits = wr <= wv
            k = hits.sum()
            hit_rates.append(k / n_window)

            if k > 0:
                # FZ-like correction: adjust model ES to match data
                corrections.append(wr[hits].mean() - we[hits].mean())
            else:
                corrections.append(0.0)

        corrections = np.array(corrections)
        hit_rates_arr = np.array(hit_rates)

        # Raw SD of correction path
        sd_raw = np.std(corrections)

        # Benchmark from true conditional VaR
        tail_resids = []
        for s in starts:
            wr = r[s:s + n_window]
            wv_true = var_true[s:s + n_window]
            mask = wr <= wv_true
            if mask.any():
                tail_resids.extend((wv_true[mask] - wr[mask]).tolist())
        sigma_tail = np.std(tail_resids) if tail_resids else 1.0
        bound = sigma_tail / np.sqrt(n_window * alpha_sim)

        all_R_raw.append(sd_raw / bound)
        all_hit.append(np.mean(hit_rates_arr))

        uc_list = []
        for hr in hit_rates_arr:
            k = int(hr * n_window)
            n0 = n_window - k
            if 0 < k < n_window:
                pi_hat = hr
                uc = 2 * (k * np.log(pi_hat / alpha_sim) +
                          n0 * np.log((1 - pi_hat) / (1 - alpha_sim)))
                uc_list.append(uc)
        all_uc.append(np.median(uc_list) if uc_list else 0.0)

    R_mean = np.mean(all_R_raw)
    hit_mean = np.mean(all_hit)
    uc_mean = np.mean(all_uc)

    results_sim.append({
        "label": delta_label,
        "delta": delta_val,
        "hit_rate": hit_mean,
        "median_uc": uc_mean,
        "R": R_mean,
    })
    print(f"  {delta_label} (delta={delta_val}): "
          f"hit={hit_mean:.4f}, UC={uc_mean:.1f}, R={R_mean:.2f}")

# LaTeX table
tex_sim = r"""\begin{tabular}{lccc}
\toprule
VaR model & Hit rate & Median UC stat & ES dispersion ratio ($R$) \\
\midrule"""
for res in results_sim:
    label = res["label"]
    if res["delta"] == 0:
        desc = "true $\\sigma_t$"
    elif res["delta"] == 1.0:
        desc = "unconditional $\\bar{\\sigma}$"
    else:
        pct = int(res["delta"] * 100)
        desc = f"{100-pct}\\% cond.\\ + {pct}\\% uncond."
    tex_sim += (f"\n  {label} ({desc}) & "
                f"{res['hit_rate']:.4f} & {res['median_uc']:.1f} & "
                f"{res['R']:.2f} \\\\")
tex_sim += r"""
\bottomrule
\end{tabular}"""
(TABDIR / "var_sim.tex").write_text(tex_sim)
print("  var_sim.tex saved")


# =====================================================================
# 4. DETRENDING ILLUSTRATION FIGURE
# =====================================================================
print("\n4. DETRENDING ILLUSTRATION FIGURE")
print("-" * 40)

sp_gjr = roll[(roll["asset"] == "SP500") &
              (roll["forecaster"] == "GJR-GARCH-t") &
              (roll["alpha"] == 0.025)].copy()
sp_gjr = sp_gjr.sort_values("date").reset_index(drop=True)

if len(sp_gjr) > 0:
    r_hat = sp_gjr["r_hat"].values
    dates = sp_gjr["date"].values
    window = 12
    ma = pd.Series(r_hat).rolling(window, center=True, min_periods=1).mean().values
    detrended = r_hat - ma

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(dates, r_hat, color=MAINBLUE, linewidth=0.7)
    axes[0].set_ylabel(r"$\hat{r}_n$")
    axes[0].set_title("(a) Raw ES correction path", loc="left", fontsize=11)
    axes[0].axhline(0, color="grey", linewidth=0.5, linestyle="--")

    axes[1].plot(dates, r_hat, color="lightgrey", linewidth=0.5, label=r"Raw $\hat{r}_n$")
    axes[1].plot(dates, ma, color=IDARED, linewidth=1.2, label="252-day MA")
    axes[1].set_ylabel(r"$\hat{r}_n$")
    axes[1].set_title("(b) Moving average overlay", loc="left", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(dates, detrended, color=FOREST, linewidth=0.7)
    axes[2].set_ylabel(r"Detrended $\hat{r}_n$")
    axes[2].set_title("(c) Detrended correction (estimation noise)", loc="left", fontsize=11)
    axes[2].axhline(0, color="grey", linewidth=0.5, linestyle="--")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(r"S&P 500, GJR-GARCH-$t$, $\alpha = 2.5\%$", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_detrend_example.pdf", **SAVE_KW)
    fig.savefig(FIGDIR / "fig_detrend_example.png", **SAVE_KW, dpi=150)
    plt.close(fig)
    print("  fig_detrend_example saved")


# =====================================================================
# 5. FOREST PLOT OF ASSET-SPECIFIC SAMPLE SIZES
# =====================================================================
print("\n5. FOREST PLOT OF ASSET-SPECIFIC SAMPLE SIZES")
print("-" * 40)

asset_ss = pd.read_csv(OUT / "tables" / "asset_sample_sizes_025.csv")
asset_ss = asset_ss.sort_values("sigma_tail", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 10))
y_pos = np.arange(len(asset_ss))
n_50bp = asset_ss["n_0.005"].values

bars = ax.barh(y_pos, n_50bp, color=MAINBLUE, alpha=0.7, height=0.7)
ax.axvline(250, color=IDARED, linewidth=1.5, linestyle="--",
           label="FRTB 250-day window")

for i, (bar, n) in enumerate(zip(bars, n_50bp)):
    if n > 250:
        bar.set_color(IDARED)
        bar.set_alpha(0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(asset_ss["asset"], fontsize=9)
ax.set_xlabel("Required calibration window $n$ (trading days)", fontsize=11)
ax.set_title(r"Required $n$ for 50 bp ES precision at $\alpha = 2.5\%$", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_forest_sample_sizes.pdf", **SAVE_KW)
fig.savefig(FIGDIR / "fig_forest_sample_sizes.png", **SAVE_KW, dpi=150)
plt.close(fig)
print("  fig_forest_sample_sizes saved")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n--- Kappa sensitivity (alpha=2.5%) ---")
for _, row in k025.iterrows():
    print(f"  kappa={row['kappa']:.1f}: {row['pct']:.1f}%")
print("\n--- Excl. Chronos-Small ---")
print(f"  All:       {all_pct:.1f}%, R={all_R_median:.2f}, rho={rho_all:.3f}")
print(f"  Excl Chr:  {excl_pct:.1f}%, R={excl_R_median:.2f}, rho={rho_excl:.3f}")
print(f"  Calib:     {calib_pct:.1f}%, R={calib_R_median:.2f}, rho={rho_calib:.3f}")
print("\n--- VaR-miscalibration simulation ---")
for res in results_sim:
    print(f"  {res['label']}: hit={res['hit_rate']:.4f}, UC={res['median_uc']:.1f}, R={res['R']:.2f}")
