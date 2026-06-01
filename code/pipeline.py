"""
Pipeline: "How Much Data Does Expected Shortfall Need?"
Computes rolling-window FZ recalibration across 24 assets x 4 forecasters x 3 alpha levels.

Key methodology:
- Rolling 250-day windows, step=21 (monthly)
- FZ_0 loss minimized by Nelder-Mead with warm-starting
- RMSE = detrended SD of r_hat (12-month MA removed to isolate estimation noise)
- Theoretical bound = sigma_tail / sqrt(n * alpha)

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t, chi2
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────
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

ALPHAS = [0.01, 0.025, 0.05]
WINDOW = 250
STEP   = 21
SMOOTH_WINDOW = 12  # 12 steps = ~252 days for detrending
SEARCH_HALF = 0.20

FORECASTERS = {
    "GJR-GARCH-t":  "gjr",
    "TimesFM-2.5":  "timesfm",
    "Chronos-Small": "chronos",
    "Moirai-2.0":   "moirai",
}


# ── data loading ───────────────────────────────────────────────────

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
        # infer Student-t df from VaR_0.01 / std ratio
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


# ── FZ loss ────────────────────────────────────────────────────────

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


# ── rolling estimation ─────────────────────────────────────────────

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

        # per-window sigma_tail
        hits = Xw <= Vw
        if hits.sum() >= 3:
            sig = np.std(Vw[hits] - Xw[hits], ddof=1)
        else:
            sig = np.nan

        results.append({"date": dates[end-1], "q_hat": q, "r_hat": r,
                         "sigma_tail_w": sig})

    return pd.DataFrame(results)


# ── detrended RMSE ─────────────────────────────────────────────────

def detrended_rmse(r_hat_series):
    s = pd.Series(r_hat_series)
    if len(s) < 2 * SMOOTH_WINDOW:
        return s.std(ddof=1)
    trend = s.rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW//2).mean()
    resid = s - trend
    return resid.dropna().std(ddof=1)


# ── Kupiec p-value ─────────────────────────────────────────────────

def kupiec_p(X, V, alpha):
    n = len(X)
    hits = (X <= V).sum()
    pi_hat = hits / n
    if pi_hat == 0 or pi_hat == 1:
        return 0.0
    lr = -2 * (n * np.log(1-alpha) + hits * np.log(alpha)
               - (n-hits) * np.log(1-pi_hat) - hits * np.log(pi_hat))
    return 1 - chi2.cdf(lr, df=1)


# ── main ───────────────────────────────────────────────────────────

def run():
    np.random.seed(42)  # deterministic FZ random restarts -> bit-exact reproducibility
    results = []
    all_rolling = []
    total = len(ASSETS) * len(FORECASTERS) * len(ALPHAS)
    i, t0 = 0, time.time()

    for asset in ASSETS:
        try:
            ret = load_returns(asset)
        except Exception as e:
            print(f"[SKIP] {asset}: {e}")
            continue

        for fname, fkey in FORECASTERS.items():
            for alpha in ALPHAS:
                i += 1
                try:
                    vhat, ehat = load_forecasts(asset, fkey, alpha)
                except Exception as e:
                    print(f"  [{i}/{total}] SKIP {asset}/{fname}/a={alpha}: {e}")
                    continue

                roll = rolling_recalib(ret, vhat, ehat, alpha)
                if len(roll) < 10:
                    print(f"  [{i}/{total}] SKIP {asset}/{fname}/a={alpha}: <10 windows")
                    continue

                # full-sample sigma_tail
                idx = ret.index.intersection(vhat.index).sort_values()
                Xf, Vf = ret.loc[idx].values, vhat.loc[idx].values
                hits = Xf <= Vf
                sigma_tail = np.std(Vf[hits] - Xf[hits], ddof=1) if hits.sum() >= 5 else np.nan
                bound = sigma_tail / np.sqrt(WINDOW * alpha) if not np.isnan(sigma_tail) else np.nan

                # metrics
                raw_sd = roll["r_hat"].std(ddof=1)
                dtrend_sd = detrended_rmse(roll["r_hat"].values)
                mean_r = roll["r_hat"].mean()

                # per-window bound (median)
                med_bound_w = roll["sigma_tail_w"].median() / np.sqrt(WINDOW * alpha)

                ratio_raw = raw_sd / bound if bound > 1e-12 else np.nan
                ratio_dtr = dtrend_sd / bound if bound > 1e-12 else np.nan
                ratio_dtr_w = dtrend_sd / med_bound_w if med_bound_w > 1e-12 else np.nan

                kup = kupiec_p(Xf, Vf, alpha)

                results.append({
                    "asset": asset, "forecaster": fname, "alpha": alpha,
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
                        "asset": asset, "forecaster": fname, "alpha": alpha,
                        "date": row["date"], "q_hat": row["q_hat"],
                        "r_hat": row["r_hat"], "sigma_tail_w": row["sigma_tail_w"],
                    })

                el = time.time() - t0
                eta = (total - i) / (i / el) if el > 0 else 0
                print(f"  [{i}/{total}] {asset}/{fname}/a={alpha}: "
                      f"dtr_sd={dtrend_sd:.4f} bound={bound:.4f} "
                      f"ratio={ratio_dtr:.2f} ({el:.0f}s, ~{eta:.0f}s left)")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "data" / "recalib_results.csv", index=False)

    dfr = pd.DataFrame(all_rolling)
    dfr.to_csv(OUT / "data" / "rolling_estimates.csv", index=False)

    print(f"\nDone: {len(df)} cells. Saved to {OUT / 'data'}")
    return df


def make_sample_size_table():
    np.random.seed(42)
    X = student_t.rvs(df=5, size=500_000)
    alphas_t = [0.005, 0.01, 0.025, 0.05]
    tolerances = [0.10, 0.20, 0.30, 0.50]

    rows = []
    for a in alphas_t:
        var_a = np.quantile(X, a)
        tail = X[X <= var_a]
        C = np.std(var_a - tail, ddof=1)
        row = {"alpha": a, "C": round(C, 4)}
        for eps in tolerances:
            row[f"eps_{eps}"] = int(np.ceil((C / eps) ** 2 / a))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "sample_size_rule.csv", index=False)

    # also make LaTeX
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$\alpha$ & $C$ & $\varepsilon=10\%$ & $20\%$ & $30\%$ & $50\%$ \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"  {r['alpha']:.1%} & {r['C']:.3f} & "
                     f"{r['eps_0.1']:,} & {r['eps_0.2']:,} & "
                     f"{r['eps_0.3']:,} & {r['eps_0.5']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "sample_size_rule.tex").write_text("\n".join(lines))

    print("Sample-size table:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("How Much Data Does Expected Shortfall Need?")
    print("=" * 60)

    print("\n[1/2] Sample-size rule...")
    make_sample_size_table()

    print("\n[2/2] Rolling FZ recalibration...")
    run()
