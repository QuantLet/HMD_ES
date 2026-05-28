"""
HS-250 naive benchmark: Historical Simulation with 250-day rolling window.
Generates VaR/ES forecasts, runs FZ recalibration, and produces comparison table.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from pathlib import Path
import time, warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────
BASE = Path("/Users/danielpele/Documents/2026 CFP LLM VaR/cfp_ijf_data")
OUT  = Path("/Users/danielpele/Documents/2026 CFP LLM ES/HowMuchData")

RETURNS_DIR = BASE / "returns"

ASSETS = [
    "SP500", "STOXX", "GDAXI", "FCHI", "FTSE100", "NIKKEI", "HSI",
    "BOVESPA", "NIFTY", "ASX200", "ICLN",
    "TLT", "IBGL",
    "DJCI", "GOLD", "WTI", "NATGAS", "CBU0",
    "BTC", "ETH",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
]

ALPHAS = [0.01, 0.025, 0.05]
WINDOW = 250
STEP = 21
SMOOTH_WINDOW = 12
SEARCH_HALF = 0.20
HS_NAME = "HS-250"


def load_returns(asset):
    df = pd.read_csv(RETURNS_DIR / f"{asset}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()["log_return"]


def generate_hs250_forecasts(returns):
    """Rolling 250-day empirical VaR and ES for all alpha levels."""
    n = len(returns)
    dates = returns.index
    vals = returns.values

    records = []
    for end in range(WINDOW, n):
        window = vals[end - WINDOW : end]
        row = {"date": dates[end]}
        for alpha in ALPHAS:
            var_a = np.quantile(window, alpha)
            tail = window[window <= var_a]
            es_a = tail.mean() if len(tail) > 0 else var_a
            row[f"VaR_{alpha}"] = var_a
            row[f"ES_{alpha}"] = es_a
        records.append(row)

    df = pd.DataFrame(records).set_index("date")
    return df


# ── FZ loss (from pipeline.py) ────────────────────────────────────

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


def rolling_recalib(returns, var_hat, es_hat, alpha):
    idx = returns.index.intersection(var_hat.index).intersection(es_hat.index).sort_values()
    X = returns.loc[idx].values
    V = var_hat.loc[idx].values
    E = es_hat.loc[idx].values
    dates = idx

    results = []
    prev_sol = [0.0, 0.0]

    for end in range(WINDOW, len(X), STEP):
        s = end - WINDOW
        Xw, Vw, Ew = X[s:end], V[s:end], E[s:end]
        if np.any(Ew >= 0) or np.any(np.isnan(Ew)) or np.any(np.isnan(Vw)):
            continue
        q, r = fit_fz(Xw, Vw, Ew, alpha, x0=prev_sol)
        prev_sol = [q, r]

        hits = Xw <= Vw
        sig = np.std(Vw[hits] - Xw[hits], ddof=1) if hits.sum() >= 3 else np.nan

        results.append({"date": dates[end - 1], "q_hat": q, "r_hat": r,
                        "sigma_tail_w": sig})

    return pd.DataFrame(results)


def detrended_rmse(r_hat_series):
    s = pd.Series(r_hat_series)
    if len(s) < 2 * SMOOTH_WINDOW:
        return s.std(ddof=1)
    trend = s.rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW // 2).mean()
    resid = s - trend
    return resid.dropna().std(ddof=1)


def kupiec_p(X, V, alpha):
    n = len(X)
    hits = (X <= V).sum()
    pi_hat = hits / n
    if pi_hat == 0 or pi_hat == 1:
        return 0.0
    lr = -2 * (n * np.log(1 - alpha) + hits * np.log(alpha)
               - (n - hits) * np.log(1 - pi_hat) - hits * np.log(pi_hat))
    return 1 - chi2.cdf(lr, df=1)


# ── main ──────────────────────────────────────────────────────────

def run():
    results = []
    all_rolling = []
    total = len(ASSETS) * len(ALPHAS)
    i, t0 = 0, time.time()

    for asset in ASSETS:
        try:
            ret = load_returns(asset)
        except Exception as e:
            print(f"[SKIP] {asset}: {e}")
            continue

        hs = generate_hs250_forecasts(ret)
        print(f"  {asset}: {len(hs)} HS-250 forecasts generated")

        for alpha in ALPHAS:
            i += 1
            var_hat = hs[f"VaR_{alpha}"]
            es_hat = hs[f"ES_{alpha}"]

            roll = rolling_recalib(ret, var_hat, es_hat, alpha)
            if len(roll) < 10:
                print(f"  [{i}/{total}] SKIP {asset}/HS-250/a={alpha}: <10 windows")
                continue

            idx = ret.index.intersection(var_hat.index).sort_values()
            Xf, Vf = ret.loc[idx].values, var_hat.loc[idx].values
            hits_mask = Xf <= Vf
            sigma_tail = np.std(Vf[hits_mask] - Xf[hits_mask], ddof=1) if hits_mask.sum() >= 5 else np.nan
            bound = sigma_tail / np.sqrt(WINDOW * alpha) if not np.isnan(sigma_tail) else np.nan

            raw_sd = roll["r_hat"].std(ddof=1)
            dtrend_sd = detrended_rmse(roll["r_hat"].values)
            mean_r = roll["r_hat"].mean()
            med_bound_w = roll["sigma_tail_w"].median() / np.sqrt(WINDOW * alpha)

            ratio_raw = raw_sd / bound if bound > 1e-12 else np.nan
            ratio_dtr = dtrend_sd / bound if bound > 1e-12 else np.nan
            ratio_dtr_w = dtrend_sd / med_bound_w if med_bound_w > 1e-12 else np.nan

            kup = kupiec_p(Xf, Vf, alpha)

            results.append({
                "asset": asset, "forecaster": HS_NAME, "alpha": alpha,
                "n_windows": len(roll),
                "r_hat_mean": mean_r,
                "raw_sd": raw_sd,
                "detrended_sd": dtrend_sd,
                "sigma_tail": sigma_tail,
                "bound": bound,
                "med_bound_w": med_bound_w,
                "ratio_raw": ratio_raw,
                "ratio_detrended": ratio_dtr,
                "ratio_detrended_w": ratio_dtr_w,
                "kupiec_p": kup,
                "n_alpha": WINDOW * alpha,
            })

            for _, row in roll.iterrows():
                all_rolling.append({
                    "asset": asset, "forecaster": HS_NAME, "alpha": alpha,
                    "date": row["date"], "q_hat": row["q_hat"],
                    "r_hat": row["r_hat"], "sigma_tail_w": row["sigma_tail_w"],
                })

            el = time.time() - t0
            eta = (total - i) / max(i / el, 1e-6)
            print(f"  [{i}/{total}] {asset}/HS-250/a={alpha}: "
                  f"dtr_sd={dtrend_sd:.4f} bound={bound:.4f} "
                  f"ratio={ratio_dtr:.2f} ({el:.0f}s, ~{eta:.0f}s left)")

    hs_df = pd.DataFrame(results)
    hs_roll = pd.DataFrame(all_rolling)

    # ── merge into existing data ──────────────────────────────────
    existing = pd.read_csv(OUT / "data" / "recalib_results.csv")
    existing = existing[existing["forecaster"] != HS_NAME]
    merged = pd.concat([existing, hs_df], ignore_index=True)
    merged.to_csv(OUT / "data" / "recalib_results.csv", index=False)

    existing_roll = pd.read_csv(OUT / "data" / "rolling_estimates.csv")
    existing_roll = existing_roll[existing_roll["forecaster"] != HS_NAME]
    merged_roll = pd.concat([existing_roll, hs_roll], ignore_index=True)
    merged_roll.to_csv(OUT / "data" / "rolling_estimates.csv", index=False)

    print(f"\nHS-250: {len(hs_df)} cells computed, merged into recalib_results.csv")

    # ── summary table ─────────────────────────────────────────────
    print("\n=== HS-250 Summary ===")
    for alpha in ALPHAS:
        sub = hs_df[hs_df["alpha"] == alpha]
        print(f"  alpha={alpha}: n={len(sub)}, "
              f"median R={sub['ratio_detrended'].median():.2f}, "
              f"mean R={sub['ratio_detrended'].mean():.2f}, "
              f"median kupiec_p={sub['kupiec_p'].median():.3f}")

    # ── comparison table (all forecasters at alpha=2.5%) ──────────
    all_data = merged[merged["alpha"] == 0.025]
    comp = all_data.groupby("forecaster").agg(
        median_R=("ratio_detrended", "median"),
        mean_R=("ratio_detrended", "mean"),
        median_kupiec=("kupiec_p", "median"),
        n_assets=("asset", "nunique"),
    ).round(3)
    print("\n=== Forecaster comparison at alpha=2.5% ===")
    print(comp.to_string())

    # ── generate LaTeX table ──────────────────────────────────────
    make_hs250_table(merged)

    return hs_df


def make_hs250_table(df):
    """Comparison table: all forecasters including HS-250 at alpha=2.5%."""
    sub = df[df["alpha"] == 0.025]
    rows = []
    for fname in ["GJR-GARCH-t", "TimesFM-2.5", "Moirai-2.0",
                   "Chronos-Small", HS_NAME]:
        fs = sub[sub["forecaster"] == fname]
        if len(fs) == 0:
            continue
        med_R = fs["ratio_detrended"].median()
        mean_r_hat = fs["r_hat_mean"].mean()
        med_kup = fs["kupiec_p"].median()
        n = len(fs)
        rows.append({
            "Forecaster": fname,
            "n_assets": n,
            "med_R": med_R,
            "mean_r": mean_r_hat,
            "med_kupiec": med_kup,
        })

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Forecaster & $n$ assets & Median $R$ & Mean $\bar{r}_n$ & Median Kupiec $p$ \\",
        r"\midrule",
    ]
    for r in rows:
        kup_str = f"{r['med_kupiec']:.3f}"
        lines.append(
            f"  {r['Forecaster']} & {r['n_assets']} & "
            f"{r['med_R']:.2f} & {r['mean_r']:.4f} & {kup_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (OUT / "tables" / "hs250_comparison.tex").write_text(tex)
    print(f"\nSaved tables/hs250_comparison.tex")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("HS-250 Naive Benchmark")
    print("=" * 60)
    run()
