"""
Non-overlapping window-length scaling experiment.
Step = n (zero overlap between adjacent windows).
GJR-GARCH-t only, alpha = 1%, n in {250, 500, 750, 1000}.
Minimum 5 non-overlapping windows per (asset, n) cell.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t, linregress
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings, time
warnings.filterwarnings("ignore")

BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data")
OUT  = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

ASSETS = [
    "SP500", "STOXX", "GDAXI", "FCHI", "FTSE100", "NIKKEI", "HSI",
    "BOVESPA", "NIFTY", "ASX200", "ICLN",
    "TLT", "IBGL",
    "DJCI", "GOLD", "WTI", "NATGAS", "CBU0",
    "BTC", "ETH",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
]

WINDOWS = [250, 500, 750, 1000]
ALPHA = 0.01
MIN_WINDOWS = 5
SEARCH_HALF = 0.20


def load_returns(asset):
    df = pd.read_csv(BASE / "returns" / f"{asset}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()["log_return"]


def _student_t_es(alpha, df_val, mean, std):
    df_val = np.clip(df_val, 2.1, 200)
    q = student_t.ppf(alpha, df_val)
    pdf_q = student_t.pdf(q, df_val)
    es_std = -(pdf_q / alpha) * ((df_val + q**2) / (df_val - 1))
    return mean + std * es_std


def load_gjr_forecasts(asset, alpha):
    fp = BASE / "benchmarks" / f"{asset}_gjr_garch.parquet"
    df = pd.read_parquet(fp)
    var_hat = df[f"VaR_{alpha}"].copy()
    mask = df["std"] > 1e-10
    ratios = (df.loc[mask, "VaR_0.01"].abs() / df.loc[mask, "std"]).median()
    best_df = 5.0
    for df_try in np.arange(3.0, 30.1, 0.5):
        if abs(abs(student_t.ppf(0.01, df_try)) - ratios) < \
           abs(abs(student_t.ppf(0.01, best_df)) - ratios):
            best_df = df_try
    es_hat = pd.Series(
        _student_t_es(alpha, best_df, df["mean"].values, df["std"].values),
        index=df.index)
    return var_hat, es_hat


def fz_loss(params, X, V_hat, E_hat, alpha):
    q, r = params
    V = V_hat + q
    E = E_hat + r
    if np.any(E >= -1e-10) or np.any(E >= V - 1e-10):
        return 1e12
    hit = (X <= V).astype(float)
    term1 = (1.0 / (alpha * E)) * hit * (X - V)
    term2 = V / E
    term3 = np.log(-E)
    return np.mean(term1 + term2 + term3 - 1.0)


def fit_fz(X, V, E, alpha, x0=None):
    if x0 is None:
        x0 = [0.0, 0.0]
    best_val, best_p = np.inf, tuple(x0)
    starts = [x0]
    for _ in range(2):
        starts.append([np.random.uniform(-SEARCH_HALF, SEARCH_HALF),
                       np.random.uniform(-SEARCH_HALF, SEARCH_HALF)])
    for s in starts:
        res = minimize(fz_loss, x0=s, args=(X, V, E, alpha),
                       method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-9})
        if res.fun < best_val:
            best_val, best_p = res.fun, tuple(res.x)
    return best_p


_ASSET_SEEDS = {a: 1000 + i for i, a in enumerate(ASSETS)}

def process_asset(asset):
    np.random.seed(_ASSET_SEEDS.get(asset, 42))
    try:
        ret = load_returns(asset)
        vhat, ehat = load_gjr_forecasts(asset, ALPHA)
    except Exception as e:
        return asset, None, str(e)

    idx = ret.index.intersection(vhat.index).intersection(ehat.index).sort_values()
    X_all = ret.loc[idx].values
    V_all = vhat.loc[idx].values
    E_all = ehat.loc[idx].values
    T = len(X_all)

    hits_full = X_all <= V_all
    sigma_tail = np.std(V_all[hits_full] - X_all[hits_full], ddof=1) \
        if hits_full.sum() >= 5 else np.nan

    rows = []
    for win in WINDOWS:
        step = win  # NON-OVERLAPPING
        n_possible = T // win

        if n_possible < MIN_WINDOWS:
            rows.append({
                "asset": asset, "n": win, "n_windows": 0,
                "n_possible": n_possible,
                "raw_sd": np.nan, "bound": np.nan,
                "sigma_tail": sigma_tail, "dropped": True,
                "drop_reason": f"<{MIN_WINDOWS} non-overlapping windows "
                               f"({n_possible} possible, T={T})"})
            continue

        r_hats = []
        n_fail = 0

        for k in range(n_possible):
            s = k * win
            e = s + win
            Xw, Vw, Ew = X_all[s:e], V_all[s:e], E_all[s:e]
            if np.any(Ew >= 0) or np.any(np.isnan(Ew)) or np.any(np.isnan(Vw)):
                n_fail += 1
                continue
            q, r = fit_fz(Xw, Vw, Ew, ALPHA)
            r_hats.append(r)

        n_valid = len(r_hats)

        if n_valid < MIN_WINDOWS:
            rows.append({
                "asset": asset, "n": win, "n_windows": n_valid,
                "n_possible": n_possible,
                "raw_sd": np.nan, "bound": np.nan,
                "sigma_tail": sigma_tail, "dropped": True,
                "drop_reason": f"<{MIN_WINDOWS} valid windows after FZ filter "
                               f"({n_valid}/{n_possible})"})
            continue

        raw_sd = np.std(r_hats, ddof=1)
        bound = sigma_tail / np.sqrt(win * ALPHA)

        rows.append({
            "asset": asset, "n": win, "n_windows": n_valid,
            "n_possible": n_possible,
            "raw_sd": raw_sd, "bound": bound,
            "sigma_tail": sigma_tail, "dropped": False,
            "drop_reason": ""})

    return asset, rows, None


def hc_se_slope(x, y):
    """OLS slope with HC1 heteroskedasticity-robust SE."""
    n = len(x)
    x_dm = x - x.mean()
    b = (x_dm * (y - y.mean())).sum() / (x_dm ** 2).sum()
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    hc1 = (n / (n - 2)) * np.sum(x_dm**2 * resid**2) / (np.sum(x_dm**2))**2
    se = np.sqrt(hc1)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, a, se, r2


def estimate_slopes(df):
    slopes = []
    for asset in sorted(df["asset"].unique()):
        sub = df[(df["asset"] == asset) & (~df["dropped"])]
        n_pts = len(sub)
        if n_pts < 3:
            slopes.append({
                "asset": asset, "slope": np.nan, "intercept": np.nan,
                "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "r2": np.nan, "n_points": n_pts,
                "contains_half": np.nan,
                "drop_reason": f"<3 valid n values ({n_pts})"})
            continue
        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)
        b, a, se, r2 = hc_se_slope(x, y)
        lo = b - 1.96 * se
        hi = b + 1.96 * se
        slopes.append({
            "asset": asset, "slope": b, "intercept": a,
            "se": se, "ci_lo": lo, "ci_hi": hi,
            "r2": r2, "n_points": n_pts,
            "contains_half": (lo <= -0.5 <= hi),
            "drop_reason": ""})
    return pd.DataFrame(slopes)


def make_comparison_histogram(no_slopes, ov_slopes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    no_sl = no_slopes.dropna(subset=["slope"])["slope"].values
    ov_sl = ov_slopes.dropna(subset=["slope"])["slope"].values

    no_med = np.median(no_sl)
    no_q1, no_q3 = np.percentile(no_sl, [25, 75])
    no_pct = np.mean(no_slopes.dropna(subset=["slope"])["contains_half"]) * 100

    ov_med = np.median(ov_sl)
    ov_q1, ov_q3 = np.percentile(ov_sl, [25, 75])
    ov_pct = np.mean(ov_slopes.dropna(subset=["slope"])["contains_half"]) * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(-1.5, 0.2, 18)
    ax.hist(ov_sl, bins=bins, color="gray", alpha=0.45, edgecolor="white",
            linewidth=0.6, label="Overlapping (step=21)")
    ax.hist(no_sl, bins=bins, color="#C8102E", alpha=0.75, edgecolor="white",
            linewidth=0.6, label="Non-overlapping (step=n)")

    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")
    ax.axvline(no_med, color="#C8102E", linewidth=2, linestyle="--",
               label=f"Non-overlap median $= {no_med:.2f}$")
    ax.axvline(ov_med, color="gray", linewidth=1.5, linestyle=":",
               label=f"Overlap median $= {ov_med:.2f}$")

    txt = (f"Non-overlapping ({len(no_sl)} assets)\n"
           f"  Median: {no_med:.2f}, IQR: [{no_q1:.2f}, {no_q3:.2f}]\n"
           f"  $-0.5$ in CI: {no_pct:.0f}%\n\n"
           f"Overlapping ({len(ov_sl)} assets)\n"
           f"  Median: {ov_med:.2f}, IQR: [{ov_q1:.2f}, {ov_q3:.2f}]\n"
           f"  $-0.5$ in CI: {ov_pct:.0f}%")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            family="monospace")

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of assets", fontsize=12)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram_Comparison.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram_Comparison.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print("Comparison histogram saved.")


def make_panel_figure(df, slopes_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_assets = ["SP500", "BTC", "NATGAS"]
    colors = {"SP500": "#003DA5", "BTC": "#C8102E", "NATGAS": "#DC143C"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, asset in zip(axes, panel_assets):
        sub = df[(df["asset"] == asset) & (~df["dropped"])]
        sl_row = slopes_df[slopes_df["asset"] == asset]
        if len(sl_row) == 0 or np.isnan(sl_row.iloc[0]["slope"]):
            ax.set_title(f"{asset} (insufficient data)", fontsize=12)
            continue
        sl_row = sl_row.iloc[0]
        c = colors[asset]

        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)

        ax.scatter(x, y, color=c, s=60, zorder=3)

        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = sl_row["intercept"] + sl_row["slope"] * x_fit
        ax.plot(x_fit, y_fit, color=c, linewidth=1.5, zorder=2,
                label=f"$\\hat{{b}} = {sl_row['slope']:.2f}$")

        y_ref = y.mean() - 0.5 * (x_fit - x.mean())
        ax.plot(x_fit, y_ref, color="gray", linewidth=1.2, linestyle="--",
                label="$b = -0.50$", zorder=1)

        ax.set_xlabel("$\\log(n)$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=11)
        ax.set_title(asset, fontsize=12)
        ci_str = f"[{sl_row['ci_lo']:.2f}, {sl_row['ci_hi']:.2f}]"
        ax.text(0.03, 0.03, f"95% CI: {ci_str}", transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Non-overlapping windows (step = n)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_NonOverlap.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_NonOverlap.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print("Panel figure saved.")


def make_table(df, slopes_df):
    merged = slopes_df.sort_values("slope", ascending=True, na_position="last")
    valid = merged.dropna(subset=["slope"])

    lines = [
        r"\begin{tabular}{lrrrrrrrrc}",
        r"\toprule",
        (r"Asset & Windows & $n{=}250$ & $n{=}500$ & $n{=}750$ "
         r"& $n{=}1000$ & $\hat{b}$ & 95\% CI & $R^2$ "
         r"& $-0.5 \in$ CI \\"),
        r"\midrule",
    ]

    for _, sl in valid.iterrows():
        asset = sl["asset"]
        sub = df[(df["asset"] == asset) & (~df["dropped"])]
        vals = {}
        nw = {}
        for n in [250, 500, 750, 1000]:
            row = sub[sub["n"] == n]
            if len(row) > 0 and not np.isnan(row.iloc[0]["raw_sd"]):
                vals[n] = f"{row.iloc[0]['raw_sd']:.4f}"
                nw[n] = int(row.iloc[0]["n_windows"])
            else:
                vals[n] = "--"
                nw[n] = 0
        total_w = sum(nw.values())
        ci = f"[{sl['ci_lo']:.2f}, {sl['ci_hi']:.2f}]"
        yn = "Y" if sl["contains_half"] else "N"
        lines.append(
            f"  {asset} & {total_w} & {vals[250]} & {vals[500]} & "
            f"{vals[750]} & {vals[1000]} & "
            f"${sl['slope']:.2f}$ & ${ci}$ & "
            f"${sl['r2']:.3f}$ & {yn} \\\\"
        )

    # dropped assets
    dropped_assets = merged[merged["slope"].isna()]
    if len(dropped_assets) > 0:
        lines.append(r"\midrule")
        for _, sl in dropped_assets.iterrows():
            lines.append(
                f"  {sl['asset']} & -- & -- & -- & -- & -- & "
                f"-- & -- & -- & --$^{{\\dagger}}$ \\\\")

    # summary row
    med = valid["slope"].median()
    q1 = valid["slope"].quantile(0.25)
    q3 = valid["slope"].quantile(0.75)
    pct = valid["contains_half"].astype(float).sum() / len(valid) * 100
    lines.append(r"\midrule")
    lines.append(
        f"  \\textit{{Median}} & & & & & & ${med:.2f}$ & "
        f"IQR $[{q1:.2f},{q3:.2f}]$ & & {pct:.0f}\\% \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]

    tex = "\n".join(lines)
    (OUT / "tables" / "window_scaling_24assets_nonoverlap.tex").write_text(tex)
    print("Non-overlap table saved.")
    return tex


if __name__ == "__main__":
    print("=" * 65)
    print("Non-overlapping window-length scaling (24 assets, GJR-GARCH-t)")
    print("=" * 65)
    t0 = time.time()

    print("\n[1/5] Running non-overlapping experiment (parallel)...")
    all_rows = []
    errors = []

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_asset, a): a for a in ASSETS}
        for fut in as_completed(futures):
            asset = futures[fut]
            try:
                name, rows, err = fut.result()
                if err:
                    errors.append((name, err))
                    print(f"  {name}: ERROR - {err}")
                elif rows:
                    all_rows.extend(rows)
                    valid = [r for r in rows if not r["dropped"]]
                    dropped = [r for r in rows if r["dropped"]]
                    drop_str = ", ".join("n=" + str(r["n"]) for r in dropped)
                    suffix = f"(dropped: {drop_str})" if dropped else ""
                    print(f"  {name:>8s}: {len(valid)}/{len(rows)} valid cells  {suffix}")
            except Exception as e:
                errors.append((asset, str(e)))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "tables" / "window_scaling_nonoverlap.csv", index=False)

    n_total = len(df)
    n_valid = (~df["dropped"]).sum()
    n_dropped = df["dropped"].sum()
    print(f"\n  Total cells: {n_total}, valid: {n_valid}, dropped: {n_dropped}")

    # Dropped cells detail
    dropped_df = df[df["dropped"]]
    if len(dropped_df) > 0:
        print(f"\n  Dropped cells ({len(dropped_df)}):")
        for _, r in dropped_df.iterrows():
            print(f"    {r['asset']:>8s}  n={r['n']:5d}  reason: {r['drop_reason']}")

    print("\n[2/5] Estimating per-asset slopes (HC1 robust SE)...")
    slopes_df = estimate_slopes(df)
    slopes_df.to_csv(OUT / "tables" / "window_scaling_nonoverlap_slopes.csv",
                     index=False)

    valid_slopes = slopes_df.dropna(subset=["slope"])
    dropped_slopes = slopes_df[slopes_df["slope"].isna()]

    for _, r in valid_slopes.sort_values("slope").iterrows():
        ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
        yn = "Y" if r["contains_half"] else "N"
        print(f"  {r['asset']:>8s}: b={r['slope']:+.3f} {ci}  "
              f"R2={r['r2']:.3f}  -0.5 in CI? {yn}  ({r['n_points']} pts)")

    if len(dropped_slopes) > 0:
        print(f"\n  Assets dropped from slope estimation:")
        for _, r in dropped_slopes.iterrows():
            print(f"    {r['asset']}: {r['drop_reason']}")

    # Summary -- cast to float to avoid object-dtype bugs
    sl = valid_slopes["slope"].values
    med = np.median(sl)
    q1, q3 = np.percentile(sl, [25, 75])
    ch = valid_slopes["contains_half"].astype(float)
    n_contains = int(ch.sum())
    pct_contains = n_contains / len(valid_slopes) * 100

    print(f"\n  === HEADLINE RESULTS ===")
    print(f"  Retained assets: {len(valid_slopes)}/24")
    print(f"  Dropped assets: {len(dropped_slopes)} "
          f"({', '.join(dropped_slopes['asset'].values)})")
    print(f"  Median slope: {med:.3f}")
    print(f"  IQR: [{q1:.3f}, {q3:.3f}]")
    print(f"  Min slope: {sl.min():.3f}, Max slope: {sl.max():.3f}")
    print(f"  -0.5 in CI: {n_contains}/{len(valid_slopes)} = {pct_contains:.0f}%")

    # Load overlapping slopes for comparison
    print("\n[3/5] Comparison with overlapping-window experiment...")
    ov_csv = OUT / "tables" / "window_scaling.csv"
    ov_df = pd.read_csv(ov_csv)
    from scipy.stats import linregress as lr
    ov_slopes_list = []
    for asset in sorted(ov_df["asset"].unique()):
        sub = ov_df[(ov_df["asset"] == asset) & (~ov_df["dropped"])]
        if len(sub) < 3:
            ov_slopes_list.append({"asset": asset, "slope": np.nan,
                                   "contains_half": np.nan})
            continue
        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)
        res = lr(x, y)
        lo, hi = res.slope - 1.96 * res.stderr, res.slope + 1.96 * res.stderr
        ov_slopes_list.append({
            "asset": asset, "slope": res.slope,
            "ci_lo": lo, "ci_hi": hi,
            "contains_half": (lo <= -0.5 <= hi)})
    ov_slopes = pd.DataFrame(ov_slopes_list)
    ov_valid = ov_slopes.dropna(subset=["slope"])
    ov_med = np.median(ov_valid["slope"].values)
    ov_q1, ov_q3 = np.percentile(ov_valid["slope"].values, [25, 75])
    ov_pct = ov_valid["contains_half"].mean() * 100

    print(f"  Overlapping:     median={ov_med:.3f}  IQR=[{ov_q1:.3f},{ov_q3:.3f}]  "
          f"-0.5 in CI: {ov_pct:.0f}%  ({len(ov_valid)} assets)")
    print(f"  Non-overlapping: median={med:.3f}  IQR=[{q1:.3f},{q3:.3f}]  "
          f"-0.5 in CI: {pct_contains:.0f}%  ({len(valid_slopes)} assets)")
    print(f"  Shift in median: {med - ov_med:+.3f} (toward -0.5: "
          f"{'YES' if abs(med - (-0.5)) < abs(ov_med - (-0.5)) else 'NO'})")

    print("\n[4/5] Making figures...")
    make_comparison_histogram(slopes_df, ov_slopes)
    make_panel_figure(df, slopes_df)

    print("\n[5/5] Making table...")
    tex = make_table(df, slopes_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min).")
    print(f"\nFiles:")
    print(f"  tables/window_scaling_nonoverlap.csv")
    print(f"  tables/window_scaling_nonoverlap_slopes.csv")
    print(f"  tables/window_scaling_24assets_nonoverlap.tex")
    print(f"  figures/HMD_SlopeHistogram_Comparison.pdf")
    print(f"  figures/HMD_WindowScaling_NonOverlap.pdf")
    if errors:
        print(f"\nErrors: {errors}")
