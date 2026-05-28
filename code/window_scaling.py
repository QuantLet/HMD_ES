"""
Addition 1: Window-length scaling experiment (all 24 assets).
Tests the (n*alpha)^{-1/2} rate by varying n explicitly.

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

WINDOWS = [250, 500, 750, 1000, 1500, 2000]
ALPHA = 0.01
STEP = 21
SEARCH_HALF = 0.20


# ── reuse existing infrastructure ──────────────────────────────────

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
        if abs(abs(student_t.ppf(0.01, df_try)) - ratios) < abs(abs(student_t.ppf(0.01, best_df)) - ratios):
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


# ── per-asset worker ───────────────────────────────────────────────

def process_asset(asset):
    np.random.seed(hash(asset) % 2**31)
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

    # full-sample sigma_tail
    hits_full = X_all <= V_all
    sigma_tail = np.std(V_all[hits_full] - X_all[hits_full], ddof=1) if hits_full.sum() >= 5 else np.nan

    rows = []
    for win in WINDOWS:
        if win >= T:
            rows.append({"asset": asset, "n": win, "n_windows": 0,
                         "raw_sd": np.nan, "bound": np.nan,
                         "sigma_tail": sigma_tail, "dropped": True,
                         "drop_reason": f"sample too short ({T} < {win})"})
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
            q, r = fit_fz(Xw, Vw, Ew, ALPHA, x0=prev_sol)
            prev_sol = [q, r]
            r_hats.append(r)

        n_windows = len(r_hats)
        fail_rate = n_fail / n_total if n_total > 0 else 1.0

        if n_windows < 10:
            rows.append({"asset": asset, "n": win, "n_windows": n_windows,
                         "raw_sd": np.nan, "bound": np.nan,
                         "sigma_tail": sigma_tail, "dropped": True,
                         "drop_reason": f"<10 valid windows ({n_windows})"})
            continue
        if fail_rate > 0.05:
            rows.append({"asset": asset, "n": win, "n_windows": n_windows,
                         "raw_sd": np.nan, "bound": np.nan,
                         "sigma_tail": sigma_tail, "dropped": True,
                         "drop_reason": f"fail rate {fail_rate:.1%} > 5%"})
            continue

        raw_sd = np.std(r_hats, ddof=1)
        bound = sigma_tail / np.sqrt(win * ALPHA)

        rows.append({"asset": asset, "n": win, "n_windows": n_windows,
                     "raw_sd": raw_sd, "bound": bound,
                     "sigma_tail": sigma_tail, "dropped": False,
                     "drop_reason": ""})

    return asset, rows, None


# ── slope estimation ───────────────────────────────────────────────

def estimate_slopes(df):
    slopes = []
    for asset in df["asset"].unique():
        sub = df[(df["asset"] == asset) & (~df["dropped"])].copy()
        if len(sub) < 3:
            slopes.append({"asset": asset, "slope": np.nan, "se": np.nan,
                           "ci_lo": np.nan, "ci_hi": np.nan, "r2": np.nan,
                           "n_points": len(sub)})
            continue
        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)
        res = linregress(x, y)
        ci_lo = res.slope - 1.96 * res.stderr
        ci_hi = res.slope + 1.96 * res.stderr
        contains_half = (ci_lo <= -0.5 <= ci_hi)
        slopes.append({
            "asset": asset, "slope": res.slope, "se": res.stderr,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "r2": res.rvalue**2,
            "intercept": res.intercept, "n_points": len(sub),
            "contains_minus_half": contains_half
        })
    return pd.DataFrame(slopes)


# ── figures ────────────────────────────────────────────────────────

def make_histogram(slopes_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = slopes_df.dropna(subset=["slope"])
    sl = valid["slope"].values
    med = np.median(sl)
    q1, q3 = np.percentile(sl, [25, 75])
    pct_contains = valid["contains_minus_half"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(sl, bins=12, range=(-1.5, 0.0), color="#003DA5", alpha=0.7,
            edgecolor="white", linewidth=0.8)
    ax.axvline(-0.5, color="black", linewidth=2, label="Theoretical $-1/2$")
    ax.axvline(med, color="#C8102E", linewidth=2, linestyle="--",
               label=f"Median $= {med:.2f}$")

    txt = (f"Median: {med:.2f}\n"
           f"IQR: [{q1:.2f}, {q3:.2f}]\n"
           f"$-0.5$ in CI: {pct_contains:.0f}%")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    ax.set_xlabel("OLS slope $\\hat{b}$", fontsize=12)
    ax.set_ylabel("Number of assets", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram.pdf",
                bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "figures" / "HMD_SlopeHistogram.png",
                bbox_inches="tight", transparent=True, dpi=150)
    plt.close()
    print(f"Histogram: median={med:.3f}, IQR=[{q1:.3f},{q3:.3f}], "
          f"-0.5 in CI: {pct_contains:.0f}%")


def make_panel_figure(df, slopes_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_assets = ["SP500", "BTC", "NATGAS"]
    colors = {"SP500": "#003DA5", "BTC": "#C8102E", "NATGAS": "#DC143C"}
    labels = {"SP500": "S\\&P 500 (equity)", "BTC": "Bitcoin (crypto)",
              "NATGAS": "Natural Gas (commodity)"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, asset in zip(axes, panel_assets):
        sub = df[(df["asset"] == asset) & (~df["dropped"])]
        sl_row = slopes_df[slopes_df["asset"] == asset].iloc[0]
        c = colors[asset]

        x = np.log(sub["n"].values.astype(float))
        y = np.log(sub["raw_sd"].values)

        ax.scatter(x, y, color=c, s=50, zorder=3)

        # OLS fit
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = sl_row["intercept"] + sl_row["slope"] * x_fit
        ax.plot(x_fit, y_fit, color=c, linewidth=1.5, zorder=2,
                label=f"$\\hat{{b}} = {sl_row['slope']:.2f}$")

        # -0.5 reference
        y_ref = y.mean() - 0.5 * (x_fit - x.mean())
        ax.plot(x_fit, y_ref, color="gray", linewidth=1.2, linestyle="--",
                label="$b = -0.50$", zorder=1)

        # theoretical bound
        sigma_tail = sub["sigma_tail"].iloc[0]
        y_bound = np.log(sigma_tail) - 0.5 * np.log(ALPHA) - 0.5 * x_fit
        ax.plot(x_fit, y_bound, color=c, linewidth=1, linestyle=":",
                alpha=0.5, label="$\\hat{C}/\\sqrt{n\\alpha}$", zorder=1)

        ax.set_xlabel("$\\log(n)$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("$\\log(\\mathrm{SD})$", fontsize=11)
        ax.set_title(f"{asset}", fontsize=12)
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


# ── table ──────────────────────────────────────────────────────────

def make_table(df, slopes_df):
    merged = slopes_df.sort_values("slope", ascending=True)
    lines = [
        r"\begin{tabular}{lrrrrrrrc}",
        r"\toprule",
        r"Asset & $n{=}250$ & $n{=}500$ & $n{=}1000$ & $n{=}2000$ "
        r"& $\hat{b}$ & 95\% CI & $R^2$ & $-0.5 \in$ CI \\",
        r"\midrule",
    ]
    for _, sl in merged.iterrows():
        asset = sl["asset"]
        sub = df[(df["asset"] == asset) & (~df["dropped"])]
        vals = {}
        for n in [250, 500, 1000, 2000]:
            row = sub[sub["n"] == n]
            if len(row) > 0 and not np.isnan(row.iloc[0]["raw_sd"]):
                vals[n] = f"{row.iloc[0]['raw_sd']:.4f}"
            else:
                vals[n] = "--"
        if np.isnan(sl["slope"]):
            lines.append(f"  {asset} & {vals[250]} & {vals[500]} & "
                         f"{vals[1000]} & {vals[2000]} & -- & -- & -- & -- \\\\")
        else:
            ci = f"[{sl['ci_lo']:.2f}, {sl['ci_hi']:.2f}]"
            yn = "Y" if sl["contains_minus_half"] else "N"
            lines.append(
                f"  {asset} & {vals[250]} & {vals[500]} & "
                f"{vals[1000]} & {vals[2000]} & "
                f"${sl['slope']:.2f}$ & ${ci}$ & "
                f"${sl['r2']:.3f}$ & {yn} \\\\"
            )
    # summary row
    valid = merged.dropna(subset=["slope"])
    med = valid["slope"].median()
    q1 = valid["slope"].quantile(0.25)
    q3 = valid["slope"].quantile(0.75)
    pct = valid["contains_minus_half"].mean() * 100
    lines.append(r"\midrule")
    lines.append(
        f"  \\textit{{Median}} & & & & & ${med:.2f}$ & "
        f"IQR $[{q1:.2f},{q3:.2f}]$ & & {pct:.0f}\\% \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (OUT / "tables" / "window_scaling_24assets.tex").write_text(tex)
    print("Table saved.")


# ── main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Addition 1: Window-length scaling (24 assets)")
    print("=" * 60)
    t0 = time.time()

    print("\n[1/4] Running scaling experiment (parallel)...")
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
                    print(f"  {name}: {len(valid)}/{len(rows)} valid cells")
            except Exception as e:
                errors.append((asset, str(e)))
                print(f"  {asset}: EXCEPTION - {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "tables" / "window_scaling.csv", index=False)
    print(f"\n  Total cells: {len(df)}, valid: {(~df['dropped']).sum()}, "
          f"dropped: {df['dropped'].sum()}")

    print("\n[2/4] Estimating slopes...")
    slopes_df = estimate_slopes(df)
    for _, r in slopes_df.iterrows():
        if not np.isnan(r["slope"]):
            ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            yn = "Y" if r["contains_minus_half"] else "N"
            print(f"  {r['asset']:>8s}: b={r['slope']:.3f} {ci} "
                  f"R²={r['r2']:.3f} -0.5 in CI? {yn}")

    print("\n[3/4] Making figures...")
    make_histogram(slopes_df)
    make_panel_figure(df, slopes_df)

    print("\n[4/4] Making table...")
    make_table(df, slopes_df)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    if errors:
        print(f"Errors: {errors}")
