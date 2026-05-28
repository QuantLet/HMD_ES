"""
Model x alpha window-length scaling robustness experiment.

This module reuses existing VaR/ES forecast files and recomputes only the
second-stage FZ recalibration over alternative calibration-window lengths.

Full outputs:
    tables/window_scaling_model_alpha_cells.csv
    tables/window_scaling_model_alpha_slopes.csv
    tables/window_scaling_model_alpha_slopes.tex
    figures/HMD_ModelAlphaScaling.pdf
    figures/HMD_ModelAlphaScaling.png

Smoke-test outputs are written under:
    smoke/window_scaling_model_alpha_<timestamp>/

Author: Daniel Traian Pele
Affiliation: Bucharest University of Economic Studies
"""

from __future__ import annotations

import argparse
import math
import time
import zlib
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pipeline import (
    ALPHAS,
    ASSETS,
    BASE,
    CHRONOS_DIR,
    FORECASTERS,
    GJR_DIR,
    MOIRAI_DIR,
    OUT,
    RETURNS_DIR,
    STEP,
    TIMESFM_DIR,
    detrended_rmse,
    fit_fz,
    load_forecasts,
    load_returns,
)


WINDOWS = [250, 500, 750, 1000, 1500]
MIN_WINDOWS = 10
MAX_FAIL_RATE = 0.05

MODEL_COLORS = {
    "GJR-GARCH-t": "#003DA5",
    "TimesFM-2.5": "#C8102E",
    "Chronos-Small": "#228B22",
    "Moirai-2.0": "#DC143C",
}
MODEL_MARKERS = {
    "GJR-GARCH-t": "o",
    "TimesFM-2.5": "s",
    "Chronos-Small": "^",
    "Moirai-2.0": "D",
}


def parse_csv_or_list(values, cast=str):
    out = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(cast(part))
    return out


def stable_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def forecast_path(asset: str, model_key: str) -> Path:
    if model_key == "gjr":
        return GJR_DIR / f"{asset}_gjr_garch.parquet"
    if model_key == "timesfm":
        return TIMESFM_DIR / f"{asset}.parquet"
    if model_key == "chronos":
        return CHRONOS_DIR / f"{asset}.parquet"
    if model_key == "moirai":
        return MOIRAI_DIR / f"{asset}.parquet"
    raise ValueError(f"Unknown model key: {model_key}")


def output_paths(tag: str | None):
    if tag:
        root = OUT / "smoke" / tag
        table_dir = root / "tables"
        figure_dir = root / "figures"
    else:
        root = OUT
        table_dir = OUT / "tables"
        figure_dir = OUT / "figures"

    return {
        "root": root,
        "cells_csv": table_dir / "window_scaling_model_alpha_cells.csv",
        "slopes_csv": table_dir / "window_scaling_model_alpha_slopes.csv",
        "slopes_tex": table_dir / "window_scaling_model_alpha_slopes.tex",
        "figure_pdf": figure_dir / "HMD_ModelAlphaScaling.pdf",
        "figure_png": figure_dir / "HMD_ModelAlphaScaling.png",
    }


def ensure_no_overwrite(paths):
    write_paths = [p for k, p in paths.items() if k != "root"]
    existing = [p for p in write_paths if p.exists()]
    if existing:
        msg = "\n".join(str(p) for p in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{msg}")


def check_inputs(assets, model_names):
    missing = []
    sources = []
    for asset in assets:
        ret_path = RETURNS_DIR / f"{asset}.csv"
        if not ret_path.exists():
            missing.append(ret_path)
        for model_name in model_names:
            model_key = FORECASTERS[model_name]
            fp = forecast_path(asset, model_key)
            sources.append(fp)
            if not fp.exists():
                missing.append(fp)
    return missing, sources


def rolling_recalib_window(returns, var_hat, es_hat, alpha, window):
    idx = returns.index.intersection(var_hat.index).intersection(es_hat.index).sort_values()
    x_all = returns.loc[idx].values
    v_all = var_hat.loc[idx].values
    e_all = es_hat.loc[idx].values
    dates = idx

    rows = []
    n_total = 0
    n_fail = 0
    prev_sol = [0.0, 0.0]

    for end in range(window, len(x_all), STEP):
        start = end - window
        xw = x_all[start:end]
        vw = v_all[start:end]
        ew = e_all[start:end]
        n_total += 1

        if np.any(ew >= 0) or np.any(np.isnan(ew)) or np.any(np.isnan(vw)):
            n_fail += 1
            continue

        try:
            q_hat, r_hat = fit_fz(xw, vw, ew, alpha, x0=prev_sol)
        except Exception:
            n_fail += 1
            continue

        prev_sol = [q_hat, r_hat]
        rows.append({"date": dates[end - 1], "q_hat": q_hat, "r_hat": r_hat})

    return pd.DataFrame(rows), len(idx), n_total, n_fail


def run_cells(assets, model_names, alphas, windows, min_windows, max_fail_rate):
    rows = []
    total = len(assets) * len(model_names) * len(alphas)
    cell = 0

    for asset in assets:
        try:
            returns = load_returns(asset)
        except Exception as exc:
            for model_name in model_names:
                for alpha in alphas:
                    for window in windows:
                        rows.append(
                            base_error_row(
                                asset, model_name, alpha, window,
                                f"returns load failed: {exc}",
                            )
                        )
            continue

        for model_name in model_names:
            model_key = FORECASTERS[model_name]
            source = forecast_path(asset, model_key)
            for alpha in alphas:
                cell += 1
                try:
                    var_hat, es_hat = load_forecasts(asset, model_key, alpha)
                except Exception as exc:
                    for window in windows:
                        rows.append(
                            base_error_row(
                                asset, model_name, alpha, window,
                                f"forecast load failed: {exc}",
                                source,
                            )
                        )
                    print(f"[{cell}/{total}] {asset}/{model_name}/alpha={alpha}: load failed")
                    continue

                for window in windows:
                    t0 = time.time()
                    np.random.seed(stable_seed(asset, model_name, alpha, window))
                    roll, n_obs, n_total, n_fail = rolling_recalib_window(
                        returns, var_hat, es_hat, alpha, window
                    )
                    fail_rate = n_fail / n_total if n_total else 1.0
                    n_valid = len(roll)
                    raw_sd = roll["r_hat"].std(ddof=1) if n_valid >= 2 else np.nan
                    dtr_sd = detrended_rmse(roll["r_hat"].values) if n_valid >= 2 else np.nan
                    dropped = False
                    reason = ""

                    if n_valid < min_windows:
                        dropped = True
                        reason = f"<{min_windows} valid windows ({n_valid})"
                    elif fail_rate > max_fail_rate:
                        dropped = True
                        reason = f"fail rate {fail_rate:.1%} > {max_fail_rate:.1%}"
                    elif not np.isfinite(dtr_sd) or dtr_sd <= 0:
                        dropped = True
                        reason = "non-positive or missing detrended SD"

                    rows.append(
                        {
                            "asset": asset,
                            "forecaster": model_name,
                            "model_key": model_key,
                            "alpha": alpha,
                            "window": window,
                            "n_obs": n_obs,
                            "n_total_windows": n_total,
                            "n_valid_windows": n_valid,
                            "n_failed_windows": n_fail,
                            "fail_rate": fail_rate,
                            "raw_sd": raw_sd,
                            "detrended_sd": dtr_sd,
                            "dropped": dropped,
                            "drop_reason": reason,
                            "forecast_file": str(source),
                            "runtime_sec": time.time() - t0,
                        }
                    )

                valid_count = sum(
                    1
                    for r in rows
                    if r["asset"] == asset
                    and r["forecaster"] == model_name
                    and r["alpha"] == alpha
                    and not r["dropped"]
                )
                print(
                    f"[{cell}/{total}] {asset}/{model_name}/alpha={alpha}: "
                    f"{valid_count}/{len(windows)} valid windows"
                )

    return pd.DataFrame(rows)


def base_error_row(asset, model_name, alpha, window, reason, source=None):
    model_key = FORECASTERS.get(model_name, "")
    return {
        "asset": asset,
        "forecaster": model_name,
        "model_key": model_key,
        "alpha": alpha,
        "window": window,
        "n_obs": np.nan,
        "n_total_windows": 0,
        "n_valid_windows": 0,
        "n_failed_windows": 0,
        "fail_rate": np.nan,
        "raw_sd": np.nan,
        "detrended_sd": np.nan,
        "dropped": True,
        "drop_reason": reason,
        "forecast_file": "" if source is None else str(source),
        "runtime_sec": np.nan,
    }


def fixed_effect_slope(group):
    valid = group[
        (~group["dropped"])
        & np.isfinite(group["detrended_sd"])
        & (group["detrended_sd"] > 0)
    ].copy()
    if valid.empty:
        return None

    counts = valid.groupby("asset")["window"].nunique()
    keep_assets = counts[counts >= 2].index
    valid = valid[valid["asset"].isin(keep_assets)].copy()
    if valid.empty:
        return None

    valid["log_n"] = np.log(valid["window"].astype(float))
    valid["log_sd"] = np.log(valid["detrended_sd"].astype(float))
    valid["x_w"] = valid["log_n"] - valid.groupby("asset")["log_n"].transform("mean")
    valid["y_w"] = valid["log_sd"] - valid.groupby("asset")["log_sd"].transform("mean")

    sxx = float(np.sum(valid["x_w"] ** 2))
    if sxx <= 0:
        return None

    slope = float(np.sum(valid["x_w"] * valid["y_w"]) / sxx)
    resid = valid["y_w"] - slope * valid["x_w"]
    ssr = float(np.sum(resid ** 2))
    n_cells = int(len(valid))
    n_assets = int(valid["asset"].nunique())
    df_resid = n_cells - n_assets - 1

    if df_resid > 0:
        sigma2 = ssr / df_resid
        se_ols = math.sqrt(sigma2 / sxx)
        tcrit = float(student_t.ppf(0.975, df_resid))
        ci_lo_ols = slope - tcrit * se_ols
        ci_hi_ols = slope + tcrit * se_ols
        contains_ols = ci_lo_ols <= -0.5 <= ci_hi_ols
    else:
        se_ols = np.nan
        ci_lo_ols = np.nan
        ci_hi_ols = np.nan
        contains_ols = np.nan

    if n_assets > 1 and df_resid > 0:
        cluster_sum = 0.0
        for _, asset_rows in valid.assign(resid=resid).groupby("asset"):
            score = float(np.sum(asset_rows["x_w"] * asset_rows["resid"]))
            cluster_sum += score**2
        finite = (n_assets / (n_assets - 1)) * ((n_cells - 1) / df_resid)
        var_cluster = finite * cluster_sum / (sxx**2)
        se_cluster = math.sqrt(max(var_cluster, 0.0))
        ci_lo_cluster = slope - tcrit * se_cluster
        ci_hi_cluster = slope + tcrit * se_cluster
        contains_cluster = ci_lo_cluster <= -0.5 <= ci_hi_cluster
    else:
        se_cluster = np.nan
        ci_lo_cluster = np.nan
        ci_hi_cluster = np.nan
        contains_cluster = np.nan

    return {
        "slope_b": slope,
        "se_ols": se_ols,
        "ci_lo_ols": ci_lo_ols,
        "ci_hi_ols": ci_hi_ols,
        "contains_minus_half_ols": contains_ols,
        "se_cluster_asset": se_cluster,
        "ci_lo_cluster": ci_lo_cluster,
        "ci_hi_cluster": ci_hi_cluster,
        "n_assets": n_assets,
        "n_valid_cells": n_cells,
        "df_resid": df_resid,
        "contains_minus_half": contains_cluster,
    }


def estimate_slopes(cells):
    rows = []
    for (forecaster, alpha), group in cells.groupby(["forecaster", "alpha"], sort=False):
        est = fixed_effect_slope(group)
        if est is None:
            rows.append(
                {
                    "forecaster": forecaster,
                    "alpha": alpha,
                    "slope_b": np.nan,
                    "se_ols": np.nan,
                    "ci_lo_ols": np.nan,
                    "ci_hi_ols": np.nan,
                    "contains_minus_half_ols": np.nan,
                    "se_cluster_asset": np.nan,
                    "ci_lo_cluster": np.nan,
                    "ci_hi_cluster": np.nan,
                    "n_assets": 0,
                    "n_valid_cells": 0,
                    "df_resid": np.nan,
                    "contains_minus_half": np.nan,
                }
            )
        else:
            rows.append({"forecaster": forecaster, "alpha": alpha, **est})
    return pd.DataFrame(rows)


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_alpha(alpha):
    return f"{100 * float(alpha):.1f}\\%"


def write_latex_table(slopes, path):
    lines = [
        r"\begin{tabular}{llrrrrrrc}",
        r"\toprule",
        (r"Forecaster & $\alpha$ & $\hat{b}$ & OLS SE & Clust. SE "
         r"& 95\% CI & Assets & Cells & $-0.5 \in$ CI \\"),
        r"\midrule",
    ]

    for _, row in slopes.iterrows():
        if pd.isna(row["ci_lo_cluster"]):
            ci = "--"
        else:
            ci = f"[{row['ci_lo_cluster']:.3f}, {row['ci_hi_cluster']:.3f}]"
        contains = row["contains_minus_half"]
        if pd.isna(contains):
            contains_txt = "--"
        else:
            contains_txt = "Y" if bool(contains) else "N"

        lines.append(
            f"{row['forecaster']} & {fmt_alpha(row['alpha'])} & "
            f"${fmt_num(row['slope_b'])}$ & ${fmt_num(row['se_ols'])}$ & "
            f"${fmt_num(row['se_cluster_asset'])}$ & ${ci}$ & {int(row['n_assets'])} & "
            f"{int(row['n_valid_cells'])} & {contains_txt} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def make_figure(slopes, path_pdf, path_png):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    valid = slopes[np.isfinite(slopes["slope_b"])].copy()
    if valid.empty:
        ax.text(0.5, 0.5, "No valid slope estimates", ha="center", va="center")
        ax.set_axis_off()
    else:
        alpha_values = sorted(valid["alpha"].unique())
        x_pos = {alpha: i for i, alpha in enumerate(alpha_values)}
        model_names = [m for m in FORECASTERS if m in set(valid["forecaster"])]
        offsets = np.linspace(-0.27, 0.27, max(len(model_names), 1))

        for offset, model_name in zip(offsets, model_names):
            sub = valid[valid["forecaster"] == model_name].sort_values("alpha")
            xs = [x_pos[a] + offset for a in sub["alpha"]]
            ys = sub["slope_b"].values
            color = MODEL_COLORS.get(model_name, "#555555")
            marker = MODEL_MARKERS.get(model_name, "o")
            yerr = None
            if np.all(np.isfinite(sub["ci_lo_cluster"])) and np.all(np.isfinite(sub["ci_hi_cluster"])):
                yerr = np.vstack([
                    ys - sub["ci_lo_cluster"].values,
                    sub["ci_hi_cluster"].values - ys,
                ])
            ax.errorbar(
                xs, ys, yerr=yerr, fmt=marker, color=color, markersize=6,
                capsize=3, linestyle="none", label=model_name,
                markeredgecolor="white", markeredgewidth=0.5,
            )

        ax.axhline(-0.5, color="black", linestyle="--", linewidth=1.1,
                   label="theoretical b = -0.5")
        ax.set_xticks([x_pos[a] for a in alpha_values])
        ax.set_xticklabels([fmt_alpha(a) for a in alpha_values])
        ax.set_xlabel("Tail level")
        ax.set_ylabel(r"Slope $\hat{b}$ in $\log(\mathrm{SD}) = a_i + b\log(n)$")
        ax.set_title("Window-length scaling by forecaster and tail level")
        ax.legend(fontsize=8, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)
        ax.grid(False)

    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(path_png, bbox_inches="tight", transparent=True, dpi=150)
    plt.close(fig)


def write_outputs(cells, slopes, paths):
    for key in ["cells_csv", "slopes_csv", "slopes_tex", "figure_pdf", "figure_png"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    cells.to_csv(paths["cells_csv"], index=False)
    slopes.to_csv(paths["slopes_csv"], index=False)
    write_latex_table(slopes, paths["slopes_tex"])
    make_figure(slopes, paths["figure_pdf"], paths["figure_png"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run model x alpha window scaling robustness experiment."
    )
    parser.add_argument("--assets", nargs="+", default=ASSETS)
    parser.add_argument("--models", nargs="+", default=list(FORECASTERS.keys()))
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS)
    parser.add_argument("--min-windows", type=int, default=MIN_WINDOWS)
    parser.add_argument("--max-fail-rate", type=float, default=MAX_FAIL_RATE)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write to a timestamped smoke directory unless --tag is provided.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    assets = parse_csv_or_list(args.assets)
    model_names = parse_csv_or_list(args.models)
    alphas = parse_csv_or_list(args.alphas, float)
    windows = parse_csv_or_list(args.windows, int)

    unknown_models = [m for m in model_names if m not in FORECASTERS]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    tag = args.tag
    if args.smoke and not tag:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"window_scaling_model_alpha_smoke_{stamp}"

    paths = output_paths(tag)
    ensure_no_overwrite(paths)

    missing, sources = check_inputs(assets, model_names)
    if missing:
        print("Missing required existing input files:")
        for path in missing:
            print(f"  {path}")
        raise FileNotFoundError("Required input forecast/return files are missing.")

    print("Reusing existing forecast files; no forecasts will be recomputed.")
    print(f"Input source files checked: {len(sources) + len(assets)}")
    print(f"Assets: {assets}")
    print(f"Models: {model_names}")
    print(f"Alphas: {alphas}")
    print(f"Windows: {windows}")

    t0 = time.time()
    cells = run_cells(
        assets=assets,
        model_names=model_names,
        alphas=alphas,
        windows=windows,
        min_windows=args.min_windows,
        max_fail_rate=args.max_fail_rate,
    )
    slopes = estimate_slopes(cells)
    write_outputs(cells, slopes, paths)
    elapsed = time.time() - t0

    print("\nOutputs:")
    for key in ["cells_csv", "slopes_csv", "slopes_tex", "figure_pdf", "figure_png"]:
        print(f"  {key}: {paths[key]}")
    print(f"\nRuntime seconds: {elapsed:.2f}")
    n_slope = int(np.isfinite(slopes["slope_b"]).sum()) if not slopes.empty else 0
    print(f"Slope regressions with finite slope: {n_slope}/{len(slopes)}")


if __name__ == "__main__":
    main()
