"""
Non-overlapping window-length scaling experiment — ALL four forecasters.
Step = n (zero overlap between adjacent windows).
Alpha = 1%, n in {250, 500, 750, 1000}.
Minimum 5 non-overlapping windows per (asset, forecaster, n) cell.

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings, time
warnings.filterwarnings("ignore")

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

WINDOWS = [250, 500, 750, 1000]
ALPHA = 0.01
MIN_WINDOWS = 5
SEARCH_HALF = 0.20


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
        df = pd.read_parquet(GJR_DIR / f"{asset}_gjr_garch.parquet")
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

    raise ValueError(model_key)


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


def process_cell(asset, forecaster_name, model_key):
    np.random.seed(_ASSET_SEEDS.get(asset, 42))
    try:
        ret = load_returns(asset)
        vhat, ehat = load_forecasts(asset, model_key, ALPHA)
    except Exception as e:
        return asset, forecaster_name, None, str(e)

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
        n_possible = T // win

        if n_possible < MIN_WINDOWS:
            rows.append({
                "asset": asset, "forecaster": forecaster_name,
                "n": win, "n_windows": 0, "n_possible": n_possible,
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
                "asset": asset, "forecaster": forecaster_name,
                "n": win, "n_windows": n_valid, "n_possible": n_possible,
                "raw_sd": np.nan, "bound": np.nan,
                "sigma_tail": sigma_tail, "dropped": True,
                "drop_reason": f"<{MIN_WINDOWS} valid windows after FZ filter "
                               f"({n_valid}/{n_possible})"})
            continue

        raw_sd = np.std(r_hats, ddof=1)
        bound = sigma_tail / np.sqrt(win * ALPHA)

        rows.append({
            "asset": asset, "forecaster": forecaster_name,
            "n": win, "n_windows": n_valid, "n_possible": n_possible,
            "raw_sd": raw_sd, "bound": bound,
            "sigma_tail": sigma_tail, "dropped": False,
            "drop_reason": ""})

    return asset, forecaster_name, rows, None


def hc_se_slope(x, y):
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
    for forecaster in sorted(df["forecaster"].unique()):
        for asset in sorted(df["asset"].unique()):
            sub = df[(df["asset"] == asset) & (df["forecaster"] == forecaster) &
                     (~df["dropped"])]
            n_pts = len(sub)
            if n_pts < 3:
                slopes.append({
                    "asset": asset, "forecaster": forecaster,
                    "slope": np.nan, "intercept": np.nan,
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
                "asset": asset, "forecaster": forecaster,
                "slope": b, "intercept": a,
                "se": se, "ci_lo": lo, "ci_hi": hi,
                "r2": r2, "n_points": n_pts,
                "contains_half": (lo <= -0.5 <= hi),
                "drop_reason": ""})
    return pd.DataFrame(slopes)


def pooled_fe_test(df):
    """Pooled fixed-effects regression: log(SD) ~ log(n) + asset FE + forecaster FE."""
    from scipy import stats as sp_stats
    ws = df[~df["dropped"]].copy()
    ws["log_n"] = np.log(ws["n"])
    ws["log_sd"] = np.log(ws["raw_sd"])

    asset_dum = pd.get_dummies(ws["asset"], drop_first=True, dtype=float)
    fcast_dum = pd.get_dummies(ws["forecaster"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(ws)), ws["log_n"].values,
                         asset_dum.values, fcast_dum.values])
    y = ws["log_sd"].values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n_obs = len(y)
    k = X.shape[1]
    mse = np.sum(resid**2) / (n_obs - k)
    cov = mse * np.linalg.inv(X.T @ X)
    se_b = np.sqrt(cov[1, 1])
    t_val = (beta[1] - (-0.5)) / se_b
    p_val = 2 * sp_stats.norm.sf(abs(t_val))
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

    return {
        "slope": beta[1], "se": se_b, "t_stat": t_val, "p_val": p_val,
        "r2": r2, "n_obs": n_obs, "n_assets": ws["asset"].nunique(),
        "n_forecasters": ws["forecaster"].nunique(),
    }


def make_table(df, slopes_df):
    """Per-forecaster summary table."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Forecaster & Assets & Median $\hat{b}$ & IQR & $-0.5 \in$ CI & $R^2$ (med) \\",
        r"\midrule",
    ]

    for fname in ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]:
        sub = slopes_df[(slopes_df["forecaster"] == fname) &
                        slopes_df["slope"].notna()]
        if len(sub) == 0:
            lines.append(f"  {fname} & 0 & --- & --- & --- & --- \\\\")
            continue
        med = sub["slope"].median()
        q1 = sub["slope"].quantile(0.25)
        q3 = sub["slope"].quantile(0.75)
        pct = sub["contains_half"].astype(float).mean() * 100
        r2_med = sub["r2"].median()
        lines.append(
            f"  {fname} & {len(sub)} & ${med:.2f}$ & "
            f"$[{q1:.2f},\\;{q3:.2f}]$ & {pct:.0f}\\% & ${r2_med:.3f}$ \\\\"
        )

    # pooled row
    all_valid = slopes_df.dropna(subset=["slope"])
    med_all = all_valid["slope"].median()
    q1_all = all_valid["slope"].quantile(0.25)
    q3_all = all_valid["slope"].quantile(0.75)
    pct_all = all_valid["contains_half"].astype(float).mean() * 100
    r2_all = all_valid["r2"].median()
    lines.append(r"\midrule")
    lines.append(
        f"  \\textit{{Pooled}} & {len(all_valid)} & ${med_all:.2f}$ & "
        f"$[{q1_all:.2f},\\;{q3_all:.2f}]$ & {pct_all:.0f}\\% & ${r2_all:.3f}$ \\\\"
    )

    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (OUT / "tables" / "window_scaling_nonoverlap_all.tex").write_text(tex)
    return tex


def make_pooled_rate_table(pfe):
    """Pooled FE rate-test table (all forecasters)."""
    from scipy import stats as sp_stats
    tex = r"""\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
  Pooled slope $\hat{b}$ & $""" + f"{pfe['slope']:.3f}" + r"""$ \\
  Standard error & $""" + f"{pfe['se']:.3f}" + r"""$ \\
  95\% CI & $[""" + f"{pfe['slope']-1.96*pfe['se']:.3f}" + r""",\;""" + f"{pfe['slope']+1.96*pfe['se']:.3f}" + r"""]$ \\
  $t$-stat ($H_0\colon b = -0.50$) & $""" + f"{pfe['t_stat']:.2f}" + r"""$ \\
  $p$-value (two-sided) & $""" + f"{pfe['p_val']:.3f}" + r"""$ \\
  $R^2$ & $""" + f"{pfe['r2']:.3f}" + r"""$ \\
  Forecasters & """ + f"{pfe['n_forecasters']}" + r""" \\
  Assets & """ + f"{pfe['n_assets']}" + r""" \\
  Observations & """ + f"{pfe['n_obs']}" + r""" \\
\bottomrule
\end{tabular}"""
    (OUT / "tables" / "pooled_rate_test.tex").write_text(tex)
    return tex


def make_histogram(slopes_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"GJR-GARCH-t": "#003DA5", "TimesFM-2.5": "#C8102E",
              "Chronos-Small": "#228B22", "Moirai-2.0": "#DC143C"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.5, 0.2, 20)

    for fname in ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]:
        sub = slopes_df[(slopes_df["forecaster"] == fname) &
                        slopes_df["slope"].notna()]
        if len(sub) == 0:
            continue
        ax.hist(sub["slope"].values, bins=bins, color=COLORS[fname],
                alpha=0.45, edgecolor="white", linewidth=0.6, label=fname)

    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")

    all_valid = slopes_df.dropna(subset=["slope"])
    med = all_valid["slope"].median()
    ax.axvline(med, color="gray", linewidth=1.5, linestyle="--",
               label=f"Pooled median $= {med:.2f}$")

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of (asset, forecaster) pairs", fontsize=12)
    ax.set_title("Non-overlapping window scaling slopes — all forecasters")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram_AllForecasters.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram_AllForecasters.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print("  Histogram saved.")


def make_panel_figure(df, slopes_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"GJR-GARCH-t": "#003DA5", "TimesFM-2.5": "#C8102E",
              "Chronos-Small": "#228B22", "Moirai-2.0": "#DC143C"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, fname in zip(axes.flat,
                         ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]):
        sub_slopes = slopes_df[(slopes_df["forecaster"] == fname) &
                               slopes_df["slope"].notna()]
        med_slope = sub_slopes["slope"].median() if len(sub_slopes) > 0 else np.nan
        pct_ci = (sub_slopes["contains_half"].astype(float).mean() * 100
                  if len(sub_slopes) > 0 else 0)

        for asset in sorted(df["asset"].unique()):
            pts = df[(df["asset"] == asset) & (df["forecaster"] == fname) &
                     (~df["dropped"])]
            if len(pts) < 2:
                continue
            x = np.log(pts["n"].values.astype(float))
            y = np.log(pts["raw_sd"].values)
            ax.plot(x, y, "o-", color=COLORS[fname], alpha=0.3, markersize=3)

        # reference line
        xlim = np.array([np.log(200), np.log(1100)])
        y_ref = -3.0 - 0.5 * (xlim - np.log(500))
        ax.plot(xlim, y_ref, "k--", lw=1.2, alpha=0.5, label="$b = -0.50$")

        ax.set_xlabel("$\\log(n)$", fontsize=10)
        ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=10)
        ax.set_title(f"{fname}  (median $\\hat{{b}} = {med_slope:.2f}$, "
                     f"$-0.5$ in CI: {pct_ci:.0f}%)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Non-overlapping window scaling by forecaster", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_AllForecasters.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_WindowScaling_AllForecasters.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print("  Panel figure saved.")


if __name__ == "__main__":
    print("=" * 70)
    print("Non-overlapping window scaling — ALL 4 forecasters x 24 assets")
    print("=" * 70)
    t0 = time.time()

    print("\n[1/5] Running non-overlapping experiment (parallel)...")
    all_rows = []
    errors = []
    tasks = [(a, fname, mkey) for a in ASSETS
             for fname, mkey in FORECASTERS.items()]

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_cell, a, fn, mk): (a, fn)
                   for a, fn, mk in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            a, fn = futures[fut]
            try:
                asset, forecaster, rows, err = fut.result()
                if err:
                    errors.append((asset, forecaster, err))
                    print(f"  [{done}/{len(tasks)}] {asset:>8s}/{forecaster}: ERROR - {err}")
                elif rows:
                    all_rows.extend(rows)
                    valid = sum(1 for r in rows if not r["dropped"])
                    print(f"  [{done}/{len(tasks)}] {asset:>8s}/{forecaster}: "
                          f"{valid}/{len(rows)} valid cells")
            except Exception as e:
                errors.append((a, fn, str(e)))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "tables" / "window_scaling_nonoverlap_all.csv", index=False)

    n_total = len(df)
    n_valid = (~df["dropped"]).sum()
    n_dropped = df["dropped"].sum()
    print(f"\n  Total cells: {n_total}, valid: {n_valid}, dropped: {n_dropped}")

    print("\n[2/5] Estimating per-(asset, forecaster) slopes (HC1 robust SE)...")
    slopes_df = estimate_slopes(df)
    slopes_df.to_csv(OUT / "tables" / "window_scaling_nonoverlap_all_slopes.csv",
                     index=False)

    for fname in ["GJR-GARCH-t", "TimesFM-2.5", "Chronos-Small", "Moirai-2.0"]:
        sub = slopes_df[(slopes_df["forecaster"] == fname) &
                        slopes_df["slope"].notna()]
        if len(sub) == 0:
            print(f"\n  {fname}: no valid slopes")
            continue
        med = sub["slope"].median()
        q1 = sub["slope"].quantile(0.25)
        q3 = sub["slope"].quantile(0.75)
        pct = sub["contains_half"].astype(float).mean() * 100
        print(f"\n  {fname}: {len(sub)} assets, median={med:.3f}, "
              f"IQR=[{q1:.3f},{q3:.3f}], -0.5 in CI: {pct:.0f}%")

    all_valid = slopes_df.dropna(subset=["slope"])
    print(f"\n  POOLED: {len(all_valid)} pairs, median={all_valid['slope'].median():.3f}, "
          f"-0.5 in CI: {all_valid['contains_half'].astype(float).mean()*100:.0f}%")

    print("\n[3/5] Pooled FE rate test (log(SD) ~ log(n) + asset FE + forecaster FE)...")
    pfe = pooled_fe_test(df)
    print(f"  slope={pfe['slope']:.3f}, SE={pfe['se']:.3f}, "
          f"t={pfe['t_stat']:.2f}, p={pfe['p_val']:.3f}, R2={pfe['r2']:.3f}")
    print(f"  ({pfe['n_forecasters']} forecasters, {pfe['n_assets']} assets, "
          f"{pfe['n_obs']} obs)")

    print("\n[4/5] Making figures...")
    make_histogram(slopes_df)
    make_panel_figure(df, slopes_df)

    print("\n[5/5] Making tables...")
    make_table(df, slopes_df)
    make_pooled_rate_table(pfe)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min).")
    print(f"\nFiles:")
    print(f"  tables/window_scaling_nonoverlap_all.csv")
    print(f"  tables/window_scaling_nonoverlap_all_slopes.csv")
    print(f"  tables/window_scaling_nonoverlap_all.tex")
    print(f"  tables/pooled_rate_test.tex")
    print(f"  figures/HMD_SlopeHistogram_AllForecasters.pdf")
    print(f"  figures/HMD_WindowScaling_AllForecasters.pdf")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
