#!/usr/bin/env python3
"""Supplementary dense Gaussian risk and trace-calibration benchmarks.

Generate full linear-model datasets and raw Gaussian sketches. No benchmark
law is used to generate fitted risks or residuals. All estimators share their
fitted model, and matched (n,d,rep) settings share features and errors.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from .common import (NumericalFailure, fit_and_evaluate, generate_data,
                         make_sketch, run_parallel, save_metadata, scalar_record,
                         seed_rng, setup_plotting)
except ImportError:
    from common import (NumericalFailure, fit_and_evaluate, generate_data,
                        make_sketch, run_parallel, save_metadata, scalar_record,
                        seed_rng, setup_plotting)


STUDY = "dense_gaussian_benchmarks"
DEFAULT_SEED = 2026092701
DEFAULT_REPS = 3000
SAMPLE_SIZES = (128, 256, 512)
RATIOS = ((0.25, 0.50), (0.50, 0.75), (0.50, 1.0))
METHODS = ("corrected", "population_sensitivity_corrected", "gcv")
KEYS = ["setting_id", "n", "d", "m", "gamma", "rho"]
MEAN_FIELDS = ("risk", *METHODS, "b_A", "t_Sigma", "observable_calibration",
               "calibration_difference", "absolute_calibration_difference",
               "unnormalized_calibration_difference", "gcv_risk_gap")
SEED_SCHEME = (
    "SHA256 keyed streams: common features/errors for matched (n,d,rep); "
    "independent raw dense Gaussian sketch stream for each (n,d,m,rep). "
    "Seed 2026092701 is separate from the three main studies.")


def cells():
    return [dict(setting_id=f"n{n}-d{round(n*gamma)}-m{round(n*rho)}",
                 n=n, d=round(n*gamma), m=round(n*rho), gamma=gamma, rho=rho)
            for n in SAMPLE_SIZES for gamma, rho in RATIOS]


def benchmarks(n, d, m):
    """Exact and proportional formulas at the realized dimension ratios."""
    gamma, rho = d / n, m / n
    risk_limit = 1 / (1-gamma) + gamma / (rho-gamma)
    gap_limit = gamma**2 / ((1-gamma) * (rho-gamma))
    df = (m-d) * (n-d)
    return dict(
        exact_mean_risk=1 + d / (n-d-1) * (1 + (n-d) / (m-d-1)),
        risk_limit=risk_limit, gcv_gap_limit=gap_limit,
        gcv_limit=risk_limit + gap_limit,
        calibration_limit=rho * gamma / ((rho-gamma) * (1-gamma)),
        exact_mean_b_A=df / m,
        exact_variance_b_A=2 * df / m**2,
        b_A_chisquare_df=df,
        exact_mean_t_Sigma=m * d / ((m-d-1) * (n-d-1)),
        exact_mean_observable_calibration=d * m / (df-2),
        exact_mean_unnormalized_calibration_difference=(
            d * (1-df / ((m-d-1) * (n-d-1)))))


def one_rep(item, seed):
    cell, rep = item
    n, d, m = cell["n"], cell["d"], cell["m"]
    base = dict(cell, rep=rep, seed=seed, sketch="dense_gaussian", noise="gaussian",
                covariance="AR(1), correlation 0.5", sigma2=1.0,
                prediction_signal=4.0, complementary_defined=bool(m < n),
                data_draw_id=f"{seed}:data:n{n}:d{d}:gaussian:rep{rep}",
                sketch_draw_id=f"{seed}:dense-benchmarks:n{n}:d{d}:m{m}:rep{rep}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            data = generate_data(n, d, rep, seed, "gaussian")
            sketch = make_sketch(n, m, "gaussian",
                                 seed_rng(seed, STUDY, "dense-sketch", n, d, m, rep),
                                 normalize=False)
            # Full-column-rank QR; no ridge, truncation, or diagnostic SVD.
            fit = fit_and_evaluate(data, sketch, diagnostics=False)
            required = ("risk", "r2", "r1", "gcv", "b", "t")
            if not all(np.isfinite(fit[key]) for key in required) or fit["b"] <= 0:
                raise NumericalFailure("Nonfinite risk/calibration output or nonpositive b_A")
            # At m=n there is no complementary subspace. Its NaN is intentional
            # and is not part of the required outputs of this study.
            result = scalar_record(fit)
            for old, new in (("r2", "corrected"), ("r1", "population_sensitivity_corrected"),
                             ("b", "b_A"), ("t", "t_Sigma"),
                             ("validation", "complementary")):
                result[new] = result.pop(old)
            result.update(base, status="ok", failure_message="")
            result["observable_calibration"] = d / result["b_A"]
            result["calibration_difference"] = result["t_Sigma"] - result["observable_calibration"]
            result["absolute_calibration_difference"] = abs(result["calibration_difference"])
            result["unnormalized_calibration_difference"] = d - result["b_A"] * result["t_Sigma"]
            result["gcv_risk_gap"] = result["gcv"] - result["risk"]
            for method in METHODS:
                error = result[method] - result["risk"]
                result[method + "_error"] = error
                result[method + "_absolute_error"] = abs(error)
                result[method + "_squared_error"] = error**2
            fields = (*MEAN_FIELDS, *(method + suffix for method in METHODS
                                      for suffix in ("_error", "_absolute_error", "_squared_error")))
            if not all(np.isfinite(result[key]) for key in fields):
                raise NumericalFailure("Nonfinite derived error or calibration statistic")
            return result
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        return dict(base, status="numerical_failure", failure_message=f"{type(exc).__name__}: {exc}",
                    **{field: np.nan for field in MEAN_FIELDS})


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    mcse = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mcse


def summarize(frame):
    """One wide row per setting; called only after every planned fit succeeds."""
    rows = []
    for key, group in frame.groupby(KEYS, sort=True):
        result = dict(zip(KEYS, key), attempted=len(group), count=len(group), unavailable=0)
        result.update(benchmarks(result["n"], result["d"], result["m"]))
        for field in MEAN_FIELDS:
            mean, mcse = mean_mcse(group[field])
            result.update({field + "_mean": mean, field + "_mcse": mcse,
                           field + "_ci_low": mean-1.96*mcse,
                           field + "_ci_high": mean+1.96*mcse})
        for method in METHODS:
            mae, mcse = mean_mcse(group[method + "_absolute_error"])
            result.update({method + "_mae": mae, method + "_mae_mcse": mcse,
                           method + "_mae_ci_low": mae-1.96*mcse,
                           method + "_mae_ci_high": mae+1.96*mcse})
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["gamma", "rho", "n"]).reset_index(drop=True)


def make_figure(summary, out):
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    means = (("risk", "#222222", "o", r"Mean $R$"),
             ("corrected", "#0072B2", "s", r"Mean $\widehat R_{\mathrm{obs}}$"),
             ("population_sensitivity_corrected", "#009E73", "D", r"Mean $\widehat R_{\mathrm{sens}}$"),
             ("gcv", "#D55E00", "^", r"Mean $\widehat R_{\mathrm{GCV}}$"))
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 6.5), sharex=True)
    for column, (gamma, rho) in enumerate(RATIOS):
        group = summary[(summary.gamma == gamma) & (summary.rho == rho)].sort_values("n")
        top, bottom = axes[:, column]
        for field, color, marker, label in means:
            top.errorbar(group.n, group[field + "_mean"], yerr=1.96*group[field + "_mcse"],
                         color=color, marker=marker, markersize=4, linewidth=1,
                         capsize=2, label=label, zorder=3)
        top.plot(group.n, group.exact_mean_risk, color="black", linewidth=1.3,
                 linestyle="--", label=r"Exact $\mathbb{E}R$", zorder=2)
        top.axhline(group.risk_limit.iloc[0], color="#666666", linestyle=":",
                    linewidth=1.5, label=r"Risk limit $\mathcal{R}_{\rm G}$", zorder=1)
        top.axhline(group.gcv_limit.iloc[0], color="#D55E00", linestyle=":",
                    linewidth=1.5, label="GCV limit", zorder=1)
        top.set_title(rf"$(\gamma,\rho)=({gamma:.2f},{rho:.2f})$", fontsize=10, pad=19)
        top.text(0, 1.025, f"({chr(97+column)})", transform=top.transAxes, fontsize=10)
        bottom.errorbar(group.n, group.absolute_calibration_difference_mean,
                        yerr=1.96*group.absolute_calibration_difference_mcse,
                        color="#0072B2", marker="o", markersize=4, linewidth=1.2, capsize=2)
        bottom.set_yscale("log")
        ticks = ((.01, .02, .03, .05), (.05, .10, .20), (.025, .05, .10))[column]
        bottom.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        bottom.yaxis.set_minor_locator(NullLocator())
        bottom.text(0, 1.025, f"({chr(100+column)})", transform=bottom.transAxes, fontsize=10)
        bottom.set_xlabel(r"Sample size $n$", fontsize=10)
        for ax in (top, bottom):
            ax.set_xscale("log")
            ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="both", which="both", labelsize=10)
            ax.grid(axis="y", alpha=.16)
    axes[0, 0].set_ylabel("Mean risk or estimate", fontsize=10)
    axes[1, 0].set_ylabel(r"Mean $|t_\Sigma-d/b_{\mathbf{A}}|$", fontsize=10)
    axes[0, 0].set_xlim(108, 605)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .015),
               ncol=3, fontsize=10, frameon=False, columnspacing=1.15, handletextpad=.4)
    fig.subplots_adjust(left=.105, right=.99, top=.86, bottom=.27,
                        wspace=.38, hspace=.52)
    folder = Path(out) / "figures"
    folder.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"dense_benchmarks.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_report(summary, out, args, counts):
    lines = ["# Dense Gaussian risk and calibration benchmarks", "",
        f"Seed: {args.seed}; repetitions per setting: {args.reps:,}. "
        f"All {counts['fits_attempted']:,} fits succeeded in nine settings.", "",
        "Gaussian linear-model data use AR(1) covariance with correlation 0.5, noise variance "
        "one, and the common sparse signal with beta' Sigma beta = 4. Raw sketch entries are "
        "independent N(0,1/m), without trace normalization. Sample sizes are 128, 256, and 512; "
        "(d/n,m/n) are (0.25,0.50), (0.50,0.75), and (0.50,1.00).", "",
        "The target is each fitted model's exact response-prediction risk. The fit uses full-rank "
        "QR without regularization or singular-value truncation. All methods share a fit. "
        "Matched (n,d,rep) settings share features and errors; sketch streams are independent. "
        "Repetitions within each setting are independent. No draws were replaced or trimmed.", "",
        "The square raw Gaussian sketch at m=n is a random weighted least-squares fit. "
        "It is not the identity-sketch control. No complementary risk estimate is defined "
        "at that boundary; its raw-file entry is intentionally missing.", "",
        "The figure contrasts empirical means with the exact finite-sample mean risk, the "
        "proportional risk limit, and the GCV limit. Its lower panels show mean absolute "
        "calibration discrepancy |t_Sigma-d/b_A|. All error bars are pointwise 95% Monte Carlo "
        "intervals (mean +/- 1.96 standard errors), without a simultaneous guarantee. "
        "The lower-panel absolute discrepancy is distinct from the signed difference of means.", "",
        "| n | gamma | rho | Mean risk (MCSE) | Exact mean risk | Risk limit | Mean absolute calibration discrepancy (MCSE) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.n} | {row.gamma:.2f} | {row.rho:.2f} | "
                     f"{row.risk_mean:.4f} ({row.risk_mcse:.4f}) | {row.exact_mean_risk:.4f} | "
                     f"{row.risk_limit:.4f} | {row.absolute_calibration_difference_mean:.5f} "
                     f"({row.absolute_calibration_difference_mcse:.5f}) |")
    lines += ["", "## Saved outputs", "",
        "- `raw.csv.gz`: every fit, draw IDs, risk estimates, exact model risk, calibration quantities, and estimation errors.",
        "- `summary.csv`: per-setting means, Monte Carlo standard errors and intervals, MAEs, and exact/proportional benchmarks.",
        "- `failures.csv`: retained failures (empty for a successful study).",
        "- `figures/dense_benchmarks.pdf` and `.png`: six-panel benchmark figure.",
        "- `metadata.json`: settings, counts, seed scheme, source hashes, and computing provenance.", ""]
    (Path(out) / "short_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.reps < 2 or args.jobs < 1:
        parser.error("Require at least two repetitions and one worker")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.csv.gz").exists() or (out / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {out}")
    start = time.perf_counter()
    catalog = cells()
    tasks = [(cell, rep) for cell in catalog for rep in range(args.reps)]
    rows = run_parallel(partial(one_rep, seed=args.seed), tasks, args.jobs)
    frame = pd.DataFrame(rows).sort_values(KEYS + ["rep"]).reset_index(drop=True)
    temporary = out / ".raw.csv.gz.tmp"
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(out / "raw.csv.gz")
    frame[frame.status != "ok"].to_csv(out / "failures.csv", index=False)
    failures = int((frame.status != "ok").sum())
    counts = dict(settings=len(catalog), repetitions_per_setting=args.reps,
                  fits_attempted=len(frame), fits_successful=len(frame)-failures,
                  numerical_failures=failures, complete_for_reporting=failures == 0,
                  unique_training_datasets=len({(c["n"], c["d"]) for c in catalog}) * args.reps,
                  catalog=catalog, trace_normalization=False)
    if failures:
        save_metadata(out, STUDY, args, counts, time.perf_counter()-start, seed_scheme=SEED_SCHEME)
        raise RuntimeError(f"{failures} dense Gaussian fits failed. Raw data, failures, and "
                           "metadata are preserved; no partial summaries or figures published.")
    summary = summarize(frame)
    summary.to_csv(out / "summary.csv", index=False)
    make_figure(summary, out)
    write_report(summary, out, args, counts)
    save_metadata(out, STUDY, args, counts, time.perf_counter()-start, seed_scheme=SEED_SCHEME)
    print(f"Dense Gaussian study complete: {len(frame):,} fits; output: {out}", flush=True)


if __name__ == "__main__":
    main()
