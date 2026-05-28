"""
Full window-length scaling experiment:
  24 assets x 4 forecasters x 3 alphas x 5 window lengths.

Reuses existing VaR/ES forecast parquet files — no foundation-model
recomputation.  Produces:
  - tables/window_scaling_full.csv          (all cells)
  - tables/window_scaling_model_alpha_slopes.tex  (slope summary)
  - figures/HMD_WindowScaling_Full.pdf      (facet figure)

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t, norm, linregress
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings, time, itertools
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────
BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data")
OUT  = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

RETURNS_DIR  = BASE / "returns"
GJR_DIR      = BASE / "benchmarks"
TIMESFM_DIR  = BASE / "timesfm25"
CHRONOS_DIR  = BASE / "chronos_small"
MOIRAI_DIR   = BASE / "moirai2"

ASSETS = [
    "SP500", "STOXX", "GDAXI", "FCHI", "FTSE100", "NIKKEI", "HSI",
    "BOVESPA", "NIFTY", "ASX200", "ICLN",
    "TLT", "IBGL",
    "DJCI", "GOLD", "WTI", "NATGAS", "CBU0",
    "BTC", "ETH",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
]

FORECASTERS = {
    "GJR-GARCH-t":   "gjr",
    "TimesFM-2.5":   "timesfm",
    "Chronos-Small":  "chronos",
    "Moirai-2.0":    "moirai",
}

ALPHAS  = [0.01, 0.025, 0.05]
WINDOWS = [250, 500, 750, 1000, 1500]
STEP    = 21
SEARCH_HALF = 0.20


# ── data loading (reused from pipeline.py) ────────────────────────

def load_returns(asset):
    df = pd.read_csv(RETURNS_DIR / f"{asset}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()["log_return"]


def _student_t_es(alpha, df_val, mean, std):
    df_val = np.clip(df_val, 2.1, 200)
    q = student_t.ppf(alpha, df_val)
    pdf_q = student_t.pdf(q, df_val)
    es_std = -(pdf_q / alpha) * ((df_val + q**2) / (df_val - 1))
    return mean + std * es_std


def load_forecasts(asset, model_key, alpha):
    var_col = f"VaR_{alpha}"

    if model_key == "gjr":
        fp = GJR_DIR / f"{asset}_gjr_garch.parquet"
        df = pd.read_parquet(fp)
        var_hat = df[var_col].copy()
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

    elif model_key == "timesfm":
        df = pd.read_parquet(TIMESFM_DIR / f"{asset}.parquet")
        var_hat = -df[var_col].abs()
        if "df_student" in df.columns:
            es_hat = pd.Series(
                _student_t_es(alpha, df["df_student"].values,
                              df["mean"].values, df["std"].values),
                index=df.index)
            es_hat = -es_hat.abs()
        else:
            z = norm.ppf(alpha)
            es_hat = -(df["mean"].abs() + df["std"] * norm.pdf(z) / alpha)
        return var_hat, es_hat

    elif model_key == "chronos":
        df = pd.read_parquet(CHRONOS_DIR / f"{asset}.parquet")
        var_hat = df[var_col].copy()
        if var_hat.median() > 0:
            var_hat = -var_hat.abs()
        es_col = f"ES_empirical_{alpha}"
        if es_col in df.columns:
            es_hat = df[es_col].copy()
            if es_hat.median() > 0:
                es_hat = -es_hat.abs()
        else:
            z = norm.ppf(alpha)
            es_hat = -(df["mean"].abs() + df["std"] * norm.pdf(z) / alpha)
        return var_hat, es_hat

    elif model_key == "moirai":
        df = pd.read_parquet(MOIRAI_DIR / f"{asset}.parquet")
        var_hat = -df[var_col].abs()
        if "df_student" in df.columns:
            es_hat = pd.Series(
                _student_t_es(alpha, df["df_student"].values,
                              df["mean"].values, df["std"].values),
                index=df.index)
            es_hat = -es_hat.abs()
        else:
            z = norm.ppf(alpha)
            es_hat = -(df["mean"].abs() + df["std"] * norm.pdf(z) / alpha)
        return var_hat, es_hat

    raise ValueError(f"Unknown model key: {model_key}")


# ── FZ loss and fitting ──────────────────────────────────────────

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


# ── per-cell worker ──────────────────────────────────────────────

def process_cell(args):
    asset, forecaster_name, model_key, alpha, windows = args
    np.random.seed(hash((asset, forecaster_name, alpha)) % 2**31)

    try:
        ret = load_returns(asset)
        vhat, ehat = load_forecasts(asset, model_key, alpha)
    except Exception as e:
        return [(asset, forecaster_name, alpha, w, 0, np.nan, np.nan,
                 True, f"load error: {e}") for w in windows]

    idx = ret.index.intersection(vhat.index).intersection(ehat.index).sort_values()
    X_all = ret.loc[idx].values
    V_all = vhat.loc[idx].values
    E_all = ehat.loc[idx].values
    T = len(X_all)

    hits_full = X_all <= V_all
    sigma_tail = np.std(V_all[hits_full] - X_all[hits_full], ddof=1) \
        if hits_full.sum() >= 5 else np.nan

    rows = []
    for win in windows:
        if win >= T:
            rows.append((asset, forecaster_name, alpha, win, 0, np.nan,
                         sigma_tail, True,
                         f"sample too short ({T} < {win})"))
            continue

        r_hats = []
        prev_sol = [0.0, 0.0]
        n_fail = 0
        n_total = 0

        for end in range(win, T, STEP):
            s = end - win
            Xw, Vw, Ew = X_all[s:end], V_all[s:end], E_all[s:end]
            n_total += 1
            if np.any(Ew >= 0) or np.any(np.isnan(Ew)) or np.any(np.isnan(Vw)):
                n_fail += 1
                continue
            q, r = fit_fz(Xw, Vw, Ew, alpha, x0=prev_sol)
            prev_sol = [q, r]
            r_hats.append(r)

        n_windows = len(r_hats)
        fail_rate = n_fail / n_total if n_total > 0 else 1.0

        if n_windows < 10:
            rows.append((asset, forecaster_name, alpha, win, n_windows,
                         np.nan, sigma_tail, True,
                         f"<10 valid windows ({n_windows})"))
            continue
        if fail_rate > 0.05:
            rows.append((asset, forecaster_name, alpha, win, n_windows,
                         np.nan, sigma_tail, True,
                         f"fail rate {fail_rate:.1%} > 5%"))
            continue

        raw_sd = np.std(r_hats, ddof=1)
        rows.append((asset, forecaster_name, alpha, win, n_windows,
                     raw_sd, sigma_tail, False, ""))

    return rows


# ── slope estimation ─────────────────────────────────────────────

def estimate_slopes(df):
    slopes = []
    for (forecaster, alpha), grp in df.groupby(["forecaster", "alpha"]):
        for asset in grp["asset"].unique():
            sub = grp[(grp["asset"] == asset) & (~grp["dropped"])]
            if len(sub) < 3:
                slopes.append({
                    "forecaster": forecaster, "alpha": alpha, "asset": asset,
                    "slope": np.nan, "se": np.nan, "ci_lo": np.nan,
                    "ci_hi": np.nan, "r2": np.nan, "n_points": len(sub),
                    "contains_half": np.nan,
                })
                continue
            x = np.log(sub["n"].values.astype(float))
            y = np.log(sub["raw_sd"].values)
            res = linregress(x, y)
            lo = res.slope - 1.96 * res.stderr
            hi = res.slope + 1.96 * res.stderr
            slopes.append({
                "forecaster": forecaster, "alpha": alpha, "asset": asset,
                "slope": res.slope, "se": res.stderr,
                "ci_lo": lo, "ci_hi": hi, "r2": res.rvalue**2,
                "intercept": res.intercept, "n_points": len(sub),
                "contains_half": (lo <= -0.5 <= hi),
            })
    return pd.DataFrame(slopes)


def summarize_slopes(sdf):
    """Summarize slopes by (forecaster, alpha): median, IQR, % containing -0.5."""
    rows = []
    for (forecaster, alpha), grp in sdf.groupby(["forecaster", "alpha"]):
        valid = grp.dropna(subset=["slope"])
        if len(valid) == 0:
            continue
        sl = valid["slope"].values
        med = np.median(sl)
        q1, q3 = np.percentile(sl, [25, 75])
        pct = valid["contains_half"].mean() * 100
        n_assets = len(valid)

        # pooled OLS slope: regress log(SD) on log(n) pooling all assets
        # (for clustered SE)
        rows.append({
            "forecaster": forecaster, "alpha": alpha,
            "n_assets": n_assets,
            "median_slope": med, "q1": q1, "q3": q3,
            "pct_contains": pct,
        })
    return pd.DataFrame(rows)


def pooled_slope_with_cluster_se(df, sdf):
    """
    For each (forecaster, alpha), run pooled OLS of log(SD) on log(n)
    with asset-clustered standard errors.
    """
    rows = []
    for (forecaster, alpha), grp in df.groupby(["forecaster", "alpha"]):
        sub = grp[~grp["dropped"]].copy()
        if len(sub) < 5:
            continue
        sub["log_n"] = np.log(sub["n"].astype(float))
        sub["log_sd"] = np.log(sub["raw_sd"])
        sub = sub.dropna(subset=["log_sd"])

        x = sub["log_n"].values
        y = sub["log_sd"].values
        n_obs = len(x)

        # OLS
        x_dm = x - x.mean()
        b = (x_dm * (y - y.mean())).sum() / (x_dm ** 2).sum()
        a = y.mean() - b * x.mean()
        resid = y - (a + b * x)

        # conventional SE
        s2 = (resid ** 2).sum() / (n_obs - 2)
        se_ols = np.sqrt(s2 / (x_dm ** 2).sum())

        # asset-clustered SE (CR1)
        clusters = sub["asset"].values
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        meat = 0.0
        for c in unique_clusters:
            mask = clusters == c
            meat += (x_dm[mask] * resid[mask]).sum() ** 2
        # CR1 correction: G/(G-1) * (n-1)/(n-k)
        cr1 = (G / (G - 1)) * ((n_obs - 1) / (n_obs - 2))
        se_cluster = np.sqrt(cr1 * meat / (x_dm ** 2).sum() ** 2)

        # R²
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        rows.append({
            "forecaster": forecaster, "alpha": alpha,
            "pooled_slope": b, "se_ols": se_ols,
            "se_cluster": se_cluster, "r2_pooled": r2,
            "n_obs": n_obs, "n_clusters": G,
        })
    return pd.DataFrame(rows)


# ── LaTeX table ──────────────────────────────────────────────────

def make_slope_table(summary_df, pooled_df):
    """Create a clean tabular environment for paper.tex."""
    merged = summary_df.merge(pooled_df,
                              on=["forecaster", "alpha"], how="left")

    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (r"Forecaster & $\alpha$ & Assets & Median $\hat{b}$ & IQR "
         r"& $-0.5\in$CI & Pooled $\hat{b}$ & SE$_{\mathrm{OLS}}$ "
         r"& SE$_{\mathrm{clust}}$ & $R^2$ \\"),
        r"\midrule",
    ]

    forecaster_order = ["GJR-GARCH-t", "TimesFM-2.5",
                        "Chronos-Small", "Moirai-2.0"]
    prev_f = None
    for f in forecaster_order:
        for alpha in ALPHAS:
            row = merged[(merged["forecaster"] == f) &
                         (merged["alpha"] == alpha)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            f_disp = f if f != prev_f else ""
            prev_f = f
            a_str = f"{alpha*100:.1f}\\%"
            iqr = f"[{r['q1']:.2f}, {r['q3']:.2f}]"
            pct = f"{r['pct_contains']:.0f}\\%"

            if pd.notna(r.get("pooled_slope")):
                pooled = f"${r['pooled_slope']:.2f}$"
                se_o = f"${r['se_ols']:.3f}$"
                se_c = f"${r['se_cluster']:.3f}$"
                r2 = f"${r['r2_pooled']:.3f}$"
            else:
                pooled = se_o = se_c = r2 = "--"

            lines.append(
                f"  {f_disp} & {a_str} & {r['n_assets']:.0f} "
                f"& ${r['median_slope']:.2f}$ & ${iqr}$ & {pct} "
                f"& {pooled} & {se_o} & {se_c} & {r2} \\\\"
            )
        if f != forecaster_order[-1]:
            lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (OUT / "tables" / "window_scaling_model_alpha_slopes.tex").write_text(tex)
    print("Slope table saved.")
    return tex


# ── figure ───────────────────────────────────────────────────────

def make_facet_figure(df, sdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forecaster_order = ["GJR-GARCH-t", "TimesFM-2.5",
                        "Chronos-Small", "Moirai-2.0"]
    colors = {"GJR-GARCH-t": "#003DA5", "TimesFM-2.5": "#228B22",
              "Chronos-Small": "#C8102E", "Moirai-2.0": "#FF8C00"}

    fig, axes = plt.subplots(len(ALPHAS), len(forecaster_order),
                             figsize=(16, 10), sharex=True, sharey="row")

    for i, alpha in enumerate(ALPHAS):
        for j, f in enumerate(forecaster_order):
            ax = axes[i, j]
            sub = df[(df["forecaster"] == f) & (df["alpha"] == alpha) &
                     (~df["dropped"])]

            for asset in sub["asset"].unique():
                a_sub = sub[sub["asset"] == asset]
                if len(a_sub) >= 2:
                    ax.plot(np.log(a_sub["n"].values.astype(float)),
                            np.log(a_sub["raw_sd"].values),
                            color=colors[f], alpha=0.25, linewidth=0.8)

            # pooled regression line
            if len(sub) >= 5:
                x_all = np.log(sub["n"].values.astype(float))
                y_all = np.log(sub["raw_sd"].values)
                mask = np.isfinite(y_all)
                if mask.sum() >= 3:
                    res = linregress(x_all[mask], y_all[mask])
                    x_fit = np.linspace(x_all.min(), x_all.max(), 50)
                    ax.plot(x_fit, res.intercept + res.slope * x_fit,
                            color=colors[f], linewidth=2,
                            label=f"$\\hat{{b}}={res.slope:.2f}$")
                    # -0.5 reference
                    y_ref = y_all[mask].mean() - 0.5 * (x_fit - x_all[mask].mean())
                    ax.plot(x_fit, y_ref, color="gray", linewidth=1.2,
                            linestyle="--", alpha=0.7)

            if i == 0:
                ax.set_title(f, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$\\alpha={alpha*100:.1f}\\%$\n$\\log(\\mathrm{{SD}})$",
                              fontsize=9)
            if i == len(ALPHAS) - 1:
                ax.set_xlabel("$\\log(n)$", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_Full.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_Full.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print("Facet figure saved.")


# ── main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Full window-scaling: 24 assets x 4 forecasters x 3 alphas x 5 windows")
    print("=" * 70)

    # Confirm: reusing existing forecast files
    for fdir in [GJR_DIR, TIMESFM_DIR, CHRONOS_DIR, MOIRAI_DIR]:
        n_files = len(list(fdir.glob("*.parquet")))
        print(f"  {fdir.name}: {n_files} parquet files found")
    print("  -> Reusing existing VaR/ES forecasts (no recomputation)\n")

    t0 = time.time()

    # Build task list
    tasks = []
    for asset in ASSETS:
        for fname, mkey in FORECASTERS.items():
            for alpha in ALPHAS:
                tasks.append((asset, fname, mkey, alpha, WINDOWS))

    print(f"[1/4] Running {len(tasks)} cells (parallel, max_workers=8)...")
    all_rows = []
    errors = []

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_cell, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            task_info = futures[fut]
            asset, fname, _, alpha, _ = task_info
            try:
                rows = fut.result()
                all_rows.extend(rows)
                valid = [r for r in rows if not r[7]]  # r[7] = dropped
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} complete")
            except Exception as e:
                errors.append((asset, fname, alpha, str(e)))
                print(f"  ERROR {asset}/{fname}/{alpha}: {e}")

    cols = ["asset", "forecaster", "alpha", "n", "n_windows",
            "raw_sd", "sigma_tail", "dropped", "drop_reason"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(OUT / "tables" / "window_scaling_full.csv", index=False)

    n_total = len(df)
    n_valid = (~df["dropped"]).sum()
    n_dropped = df["dropped"].sum()
    print(f"\n  Total cells: {n_total}, valid: {n_valid}, dropped: {n_dropped}")

    # Valid cells by forecaster and alpha
    print("\n  Valid cells by forecaster and alpha:")
    for f in FORECASTERS:
        for alpha in ALPHAS:
            sub = df[(df["forecaster"] == f) & (df["alpha"] == alpha)]
            nv = (~sub["dropped"]).sum()
            print(f"    {f:16s}  alpha={alpha:.3f}  valid={nv}/{len(sub)}")

    # Dropped cells
    dropped = df[df["dropped"]]
    if len(dropped) > 0:
        print(f"\n  Dropped cells ({len(dropped)} total):")
        for reason, grp in dropped.groupby("drop_reason"):
            assets = sorted(grp["asset"].unique())
            print(f"    {reason}: {len(grp)} cells "
                  f"({', '.join(assets[:5])}{'...' if len(assets) > 5 else ''})")

    print("\n[2/4] Estimating per-asset slopes...")
    sdf = estimate_slopes(df)
    sdf.to_csv(OUT / "tables" / "window_scaling_full_slopes.csv", index=False)

    print("\n[3/4] Computing summary and pooled slopes...")
    summary = summarize_slopes(sdf)
    pooled = pooled_slope_with_cluster_se(df, sdf)

    for _, r in summary.iterrows():
        p = pooled[(pooled["forecaster"] == r["forecaster"]) &
                   (pooled["alpha"] == r["alpha"])]
        if len(p) > 0:
            p = p.iloc[0]
            print(f"  {r['forecaster']:16s}  alpha={r['alpha']:.3f}  "
                  f"median={r['median_slope']:.3f}  "
                  f"pooled={p['pooled_slope']:.3f} "
                  f"(SE_ols={p['se_ols']:.3f}, SE_clust={p['se_cluster']:.3f})  "
                  f"-0.5 in CI: {r['pct_contains']:.0f}%")

    tex = make_slope_table(summary, pooled)
    print("\n  Table preview:")
    print(tex)

    print("\n[4/4] Making facet figure...")
    make_facet_figure(df, sdf)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed/60:.1f} minutes ({elapsed:.0f}s).")
    print(f"Files:")
    print(f"  tables/window_scaling_full.csv")
    print(f"  tables/window_scaling_full_slopes.csv")
    print(f"  tables/window_scaling_model_alpha_slopes.tex")
    print(f"  figures/HMD_WindowScaling_Full.pdf")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
