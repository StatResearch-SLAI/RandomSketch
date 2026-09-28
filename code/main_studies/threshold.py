#!/usr/bin/env python3
"""Compare the two risk estimates using simulated linear-model data.

Every replicate generates X and y and fits the sketched regression. The
chi-square representations used in the proofs are not simulation inputs.
The five sketch sizes within a replicate share data and a row permutation;
the central setting also shares its streams with the accuracy study.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from common import (
    DEFAULT_REPS,
    DEFAULT_SEED,
    NumericalFailure,
    fit_and_evaluate,
    generate_data,
    row_sketch,
    run_parallel,
    save_metadata,
    setup_plotting,
)


SAMPLE_SIZES = (128, 256, 512)
GAMMAS = (0.25, 0.50)
OFFSETS = (-0.08, -0.04, 0.0, 0.04, 0.08)


def asymptotic_threshold(gamma):
    return 3.0 * gamma / (1.0 + 2.0 * gamma)


def sketch_sizes(n, d):
    """Five integer sketch dimensions centered at the limiting threshold."""
    rho_star = asymptotic_threshold(d / n)
    sizes = tuple(round(n * (rho_star + offset)) for offset in OFFSETS)
    if len(set(sizes)) != len(OFFSETS):
        raise ValueError(f"Rounding gives duplicate sketch sizes: {sizes}")
    if any(not d + 3 < m < n for m in sizes):
        raise ValueError(f"Need d+3 < m < n for every size: n={n}, d={d}, {sizes}")
    return sizes


def exact_mses(n, d, m, sigma2=1.0):
    """The manuscript's exact MSE formulas, including continuous m values.

    sigma2 is the noise variance, so the common scale below is sigma2**2.
    The formulas apply to any positive definite feature covariance here.
    """
    m = np.asarray(m, dtype=float)
    if np.any(m <= d + 3) or np.any(m >= n):
        raise ValueError("Exact MSEs require d+3 < m < n")
    if sigma2 <= 0:
        raise ValueError("Noise variance must be positive")
    k, q = m - d, n - m
    e = (m - 1) * (m - 3) / ((k - 1) * (k - 3))
    a = (2 * m**2 / k**3
         + 2 * d * (m - 1) / ((k - 1)**2 * (k - 3))
         + d**2 / (k**2 * (k - 1)**2))
    complementary = 2 * sigma2**2 * e / q
    corrected = sigma2**2 * (2 * q * e + m**2 * a) / n**2
    return complementary, corrected


def exact_ratio(n, d, m):
    complementary, corrected = exact_mses(n, d, m)
    return corrected / complementary


def continuous_crossover(n, d):
    """Numerical root of the exact finite-n equality near rho_*(d/n).

    This continuous root is distinct from the first-order approximation
    rho_*(d/n)-5/(12*n), and integer sketch sizes remain the simulation units.
    """
    center = asymptotic_threshold(d / n)
    lower = max((d + 3 + 1e-6) / n, center - 0.08)
    upper = min(1 - 1e-6 / n, center + 0.08)
    return float(brentq(lambda rho: exact_ratio(n, d, n * rho) - 1,
                        lower, upper, xtol=1e-14, rtol=1e-14))


def _base_record(n, d, m, rep, seed):
    return {
        "study": "threshold",
        "n": n, "d": d, "m": m, "gamma": d / n, "rho": m / n,
        "noise": "gaussian", "sketch": "row", "rep": rep,
        "seed": seed,
        "design_draw_id": f"{seed}:features:n{n}:d{d}:rep{rep}",
        "data_draw_id": f"{seed}:data:n{n}:d{d}:gaussian:rep{rep}",
        "sketch_draw_id": f"{seed}:sketch:n{n}:d{d}:m{m}:row:rep{rep}",
        "row_order_id": f"{seed}:n{n}:d{d}:rep{rep}:row",
        "fit_id": f"{seed}:n{n}:d{d}:rep{rep}:gaussian:row:m{m}",
        "covariance": "AR(1), correlation 0.5", "sigma2": 1.0,
        "prediction_signal": 4.0,
        "central_setting_shared_with_accuracy":
            m == round(n * asymptotic_threshold(d / n)),
        "status": "numerical_failure", "failure_type": "", "failure_message": "",
        "risk": np.nan,
        "complementary": np.nan, "corrected": np.nan,
        "complementary_error": np.nan, "corrected_error": np.nan,
        "complementary_squared_error": np.nan,
        "corrected_squared_error": np.nan,
    }


def simulate_replicate(item):
    """One full dataset and five fits; retain failed draws without replacement."""
    n, d, rep, seed = item
    sizes = sketch_sizes(n, d)
    numerical_errors = (NumericalFailure, np.linalg.LinAlgError, FloatingPointError)
    try:
        data = generate_data(n, d, rep, seed, noise="gaussian")
    except numerical_errors as exc:
        records = [_base_record(n, d, m, rep, seed) for m in sizes]
        for record in records:
            record.update(failure_type=type(exc).__name__, failure_message=str(exc))
        return records

    records = []
    for m in sizes:
        record = _base_record(n, d, m, rep, seed)
        try:
            sketch = row_sketch(n, m, d, rep, seed)
            result = fit_and_evaluate(data, sketch,
                                      compute_r1=False, diagnostics=False)
            risk = float(result["risk"])
            complementary = float(result["validation"])
            corrected = float(result["r2"])
            values = np.array([risk, complementary, corrected])
            if not np.all(np.isfinite(values)):
                raise NumericalFailure("Nonfinite risk or risk estimate")
            if np.any(values < 0):
                raise NumericalFailure("Negative risk or risk estimate")
            err_complementary = complementary - risk
            err_corrected = corrected - risk
            with np.errstate(over="raise", invalid="raise"):
                squared_errors = np.square([err_complementary, err_corrected])
            if not np.all(np.isfinite(squared_errors)):
                raise NumericalFailure("Nonfinite squared estimation error")
            record.update(
                status="ok", risk=risk,
                complementary=complementary, corrected=corrected,
                complementary_error=err_complementary,
                corrected_error=err_corrected,
                complementary_squared_error=float(squared_errors[0]),
                corrected_squared_error=float(squared_errors[1]),
            )
        except numerical_errors as exc:
            record.update(failure_type=type(exc).__name__, failure_message=str(exc))
        records.append(record)
    return records


def summarize(raw, repetitions):
    """Pointwise 95% paired log-delta intervals for the MSE ratio.

    A failed cell is reported but is not summarized from selected successful
    draws. Both squared errors in the influence function belong to the same
    fit and the same exact conditional risk target.
    """
    rows = []
    z975 = norm.ppf(0.975)
    for (n, d, m), group in raw.groupby(["n", "d", "m"], sort=True):
        n, d, m = int(n), int(d), int(m)
        completed = int(group["status"].eq("ok").sum())
        failed = int(len(group) - completed)
        exact_comp, exact_corr = exact_mses(n, d, m)
        row = {
            "n": n, "d": d, "m": m, "gamma": d / n, "rho": m / n,
            "repetitions_requested": repetitions,
            "repetitions_attempted": len(group),
            "repetitions_completed": completed,
            "repetitions_failed": failed,
            "status": "ok" if failed == 0 and len(group) == repetitions else "incomplete",
            "rho_star": asymptotic_threshold(d / n),
            "rho_exact_n": continuous_crossover(n, d),
            "rho_first_order_n": asymptotic_threshold(d / n) - 5 / (12 * n),
            "mse_complementary_exact": float(exact_comp),
            "mse_corrected_exact": float(exact_corr),
            "mse_ratio_exact": float(exact_corr / exact_comp),
            "mse_complementary_mc": np.nan, "mse_corrected_mc": np.nan,
            "mse_ratio_mc": np.nan, "mse_ratio_log_se": np.nan,
            "mse_ratio_ci_low": np.nan, "mse_ratio_ci_high": np.nan,
        }
        if row["status"] == "ok":
            square_comp = group["complementary_squared_error"].to_numpy()
            square_corr = group["corrected_squared_error"].to_numpy()
            mse_comp, mse_corr = float(square_comp.mean()), float(square_corr.mean())
            if min(mse_comp, mse_corr) <= 0:
                raise ValueError("A log-ratio interval requires positive empirical MSEs")
            ratio = mse_corr / mse_comp
            influence = square_corr / mse_corr - square_comp / mse_comp
            log_se = float(np.std(influence, ddof=1) / np.sqrt(completed))
            row.update(
                mse_complementary_mc=mse_comp, mse_corrected_mc=mse_corr,
                mse_ratio_mc=ratio, mse_ratio_log_se=log_se,
                mse_ratio_ci_low=ratio * np.exp(-z975 * log_se),
                mse_ratio_ci_high=ratio * np.exp(z975 * log_se),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(summary, output):
    setup_plotting()
    import matplotlib.pyplot as plt

    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.9), sharey=True)
    for row_i, gamma in enumerate(GAMMAS):
        for col_i, n in enumerate(SAMPLE_SIZES):
            ax = axes[row_i, col_i]
            d = round(n * gamma)
            group = summary[(summary["n"] == n) & (summary["d"] == d)]
            group = group.sort_values("rho")
            rho = group["rho"].to_numpy()
            ratio = group["mse_ratio_mc"].to_numpy()
            low = group["mse_ratio_ci_low"].to_numpy()
            high = group["mse_ratio_ci_high"].to_numpy()
            ax.errorbar(rho, ratio, yerr=[ratio - low, high - ratio],
                        fmt="o", markersize=4, capsize=2.5,
                        color="#0072B2", label="Data simulation", zorder=4)
            xlo = max((d + 3 + 1e-3) / n, float(rho.min()) - 0.015)
            xhi = min(1 - 1e-3 / n, float(rho.max()) + 0.015)
            grid = np.linspace(xlo, xhi, 400)
            ax.plot(grid, exact_ratio(n, d, n * grid), color="#009E73",
                    linewidth=1.5, label="Exact ratio")
            ax.axvline(asymptotic_threshold(gamma), color="#222222",
                       linestyle="--", linewidth=1.2,
                       label=r"$\rho_*(\gamma)$")
            ax.axvline(continuous_crossover(n, d), color="#D55E00",
                       linestyle=":", linewidth=1.8,
                       label=r"$\rho_n^{\mathrm{exact}}$")
            ax.axhline(1, color="#777777", linestyle="-.", linewidth=0.8)
            ax.set_title(fr"$\gamma={gamma:.2f}$, $n={n}$", fontsize=10)
            ax.set_xlabel(r"$\rho=m/n$")
            ax.set_xlim(xlo, xhi)
            ax.grid(alpha=0.2)
            if col_i == 0:
                ax.set_ylabel(r"$\mathrm{MSE}(\widehat R_{\mathrm{obs}}) / "
                              r"\mathrm{MSE}(\widehat R_{\perp,\mathbf{S}})$", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Matplotlib orders the errorbar container after the line artists.
    order = [labels.index(label) for label in
             ("Data simulation", "Exact ratio", r"$\rho_*(\gamma)$",
              r"$\rho_n^{\mathrm{exact}}$")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(folder / "threshold.pdf")
    fig.savefig(folder / "threshold.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "raw.csv.gz").exists() or (args.output / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {args.output}")
    start = time.perf_counter()
    items = [(n, round(n * gamma), rep, args.seed)
             for n in SAMPLE_SIZES for gamma in GAMMAS
             for rep in range(args.reps)]
    nested_records = run_parallel(simulate_replicate, items, args.jobs)
    raw = pd.DataFrame([record for records in nested_records for record in records])
    raw = raw.sort_values(["n", "d", "m", "rep"]).reset_index(drop=True)
    raw.to_csv(args.output / "raw.csv.gz", index=False)
    summary = summarize(raw, args.reps)
    summary.to_csv(args.output / "summary.csv", index=False)
    failed = int(raw["status"].ne("ok").sum())
    counts = {
        "settings": len(SAMPLE_SIZES) * len(GAMMAS) * len(OFFSETS),
        "independent_datasets": len(items),
        "attempted_fits": len(raw),
        "completed_fits": int(len(raw) - failed),
        "failed_fits": failed,
        "repetitions_per_setting": args.reps,
        "central_fits_shared_with_accuracy":
            len(SAMPLE_SIZES) * len(GAMMAS) * args.reps,
        "pairing": "Five m values share each dataset and nested row permutation.",
        "ratio_interval": "Pointwise 95% paired log-delta interval.",
        "scope": "Data simulation assesses ordering; it does not resolve the O(1/n) threshold shift.",
    }
    save_metadata(args.output, "threshold", args, counts, time.perf_counter() - start)
    if failed:
        raise RuntimeError(f"Preserved {failed} failed fits in raw.csv.gz; no draws were replaced")
    make_figure(summary, args.output)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(raw)} fits from {len(items)} datasets to {args.output}", flush=True)


if __name__ == "__main__":
    main()
