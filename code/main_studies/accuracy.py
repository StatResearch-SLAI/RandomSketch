#!/usr/bin/env python3
"""Accuracy of the main corrected risk estimates in a common simulation design.

The 24 settings cross three sample sizes, two dimension ratios, two sketches,
and two error distributions. Each replicate supplies the same data and sketch
to every criterion. The two sketches share data, and the two error distributions
share both the Gaussian feature draw and the sketch. Numerical failures remain
in the raw output; they are never replaced or trimmed.
"""
import argparse
from functools import partial
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd

from common import (DEFAULT_REPS, DEFAULT_SEED, NumericalFailure,
                    fit_and_evaluate, generate_data, row_sketch, run_parallel,
                    save_metadata, scalar_record, setup_plotting,
                    weighted_sketch)


METHODS = ("corrected", "population_sensitivity_corrected", "gcv", "complementary")
PLOT_METHODS = METHODS[:3]
LEGACY_FIELDS = {"r2": "corrected", "r1": "population_sensitivity_corrected",
                 "gcv": "gcv", "validation": "complementary"}
LABELS = {"corrected": r"Corrected $\widehat R_{\mathrm{obs}}$",
          "population_sensitivity_corrected": r"Population-sensitivity $\widehat R_{\mathrm{sens}}$",
          "gcv": r"GCV $\widehat R_{\mathrm{GCV}}$",
          "complementary": r"Complementary $\widehat R_{\perp,\mathbf{S}}$"}
COLORS = {"corrected": "#0072B2", "population_sensitivity_corrected": "#009E73",
          "gcv": "#D55E00", "complementary": "#CC79A7"}
MARKERS = {"corrected": "o", "population_sensitivity_corrected": "s", "gcv": "^"}
GROUP_KEYS = ["n", "d", "m", "gamma", "rho", "sketch", "noise"]
SAMPLE_SIZES = (128, 256, 512)
RATIOS = ((0.25, 0.50), (0.50, 0.75))
SKETCHES = ("row", "weighted_haar")
NOISES = ("gaussian", "rademacher")


def cells():
    return [dict(n=n, d=round(n * gamma), m=round(n * rho),
                 gamma=gamma, rho=rho)
            for n in SAMPLE_SIZES for gamma, rho in RATIOS]


def _base_record(cell, rep, seed, sketch, noise):
    n, d, m = cell["n"], cell["d"], cell["m"]
    return dict(cell, sketch=sketch, noise=noise, rep=rep, seed=seed,
                design_draw_id=f"{seed}:features:n{n}:d{d}:rep{rep}",
                data_draw_id=f"{seed}:data:n{n}:d{d}:{noise}:rep{rep}",
                sketch_draw_id=f"{seed}:sketch:n{n}:d{d}:m{m}:{sketch}:rep{rep}",
                covariance="AR(1), correlation 0.5", sigma2=1.0,
                prediction_signal=4.0)


def _failure_record(base, exc):
    return dict(base, status="numerical_failure", failure_message=str(exc),
                risk=np.nan, **{method: np.nan for method in METHODS})


def one_rep(item, seed):
    """Four paired fits per task; independently keyed tasks can run in any order."""
    cell, rep = item
    n, d, m = cell["n"], cell["d"], cell["m"]
    data = {noise: generate_data(n, d, rep, seed, noise) for noise in NOISES}
    rows = []
    for family in SKETCHES:
        try:
            sketch = (row_sketch(n, m, d, rep, seed) if family == "row"
                      else weighted_sketch(n, m, d, rep, seed))
        except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
            rows.extend(_failure_record(_base_record(cell, rep, seed, family, noise), exc)
                        for noise in NOISES)
            continue
        for noise in NOISES:
            base = _base_record(cell, rep, seed, family, noise)
            try:
                # The Gaussian model is full column rank almost surely, with
                # m-d >= 32 here. Use the same full-rank QR solve as the threshold
                # study without an additional diagnostic SVD on every fit.
                fit = fit_and_evaluate(data[noise], sketch, diagnostics=False)
                if not all(np.isfinite(fit[key])
                           for key in ("risk", "r1", "r2", "gcv", "validation")):
                    raise NumericalFailure("Nonfinite risk estimate or exact model risk")
                record = scalar_record(fit)
                # Only this adapter uses legacy names. Saved data use method names.
                for old, new in LEGACY_FIELDS.items():
                    record[new] = record.pop(old)
                record.update(base, status="ok", failure_message="")
                for method in METHODS:
                    error = record[method] - record["risk"]
                    record[method + "_error"] = error
                    record[method + "_absolute_error"] = abs(error)
                    record[method + "_squared_error"] = error ** 2
                rows.append(record)
            except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
                rows.append(_failure_record(base, exc))
    return rows


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    size = len(values)
    mean = float(values.mean()) if size else np.nan
    se = float(values.std(ddof=1) / np.sqrt(size)) if size > 1 else np.nan
    return mean, se


def summaries(frame):
    """Pointwise Monte Carlo intervals, preserving attempted/unavailable counts."""
    metrics, comparisons = [], []
    for key, group in frame.groupby(GROUP_KEYS, sort=True, dropna=False):
        base = dict(zip(GROUP_KEYS, key))
        methods = METHODS if base["noise"] == "gaussian" else PLOT_METHODS
        valid = group[group.status == "ok"]
        for method in methods:
            error = (valid[method] - valid.risk).to_numpy(dtype=float)
            error = error[np.isfinite(error)]
            bias, bias_se = _mean_mcse(error)
            mae, mae_se = _mean_mcse(np.abs(error))
            mse, mse_se = _mean_mcse(error ** 2)
            rmse = float(np.sqrt(mse))
            rmse_se = mse_se / (2 * rmse) if rmse > 0 else np.nan
            metrics.append(dict(base, method=method, attempted=len(group), count=len(error),
                unavailable=len(group) - len(error), bias=bias, bias_mcse=bias_se,
                bias_ci_low=bias - 1.96 * bias_se, bias_ci_high=bias + 1.96 * bias_se,
                mae=mae, mae_mcse=mae_se, mae_ci_low=mae - 1.96 * mae_se,
                mae_ci_high=mae + 1.96 * mae_se, mse=mse, mse_mcse=mse_se,
                rmse=rmse, rmse_mcse=rmse_se,
                scaled_mae=np.sqrt(base["n"]) * mae,
                scaled_mae_mcse=np.sqrt(base["n"]) * mae_se))
        for first, second in combinations(methods, 2):
            pair = valid[["risk", first, second]].replace([np.inf, -np.inf], np.nan).dropna()
            difference = ((pair[first] - pair.risk).abs() -
                          (pair[second] - pair.risk).abs()).to_numpy()
            mean, se = _mean_mcse(difference)
            comparisons.append(dict(base, first_method=first, second_method=second,
                attempted=len(group), count=len(pair), unavailable=len(group) - len(pair),
                paired_mae_difference=mean, difference_mcse=se,
                ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se))
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def make_figure(summary, out):
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    # Preserve both corrected curves when their MAEs nearly coincide.
    styles = {
        "corrected": dict(linestyle="-", markersize=4.2, linewidth=1.6),
        "population_sensitivity_corrected": dict(
            linestyle=(0, (5, 3)), markersize=5.8, linewidth=1.3,
            markerfacecolor="none", markeredgewidth=1.1),
        "gcv": dict(linestyle="-.", markersize=4.5, linewidth=1.25),
    }
    rows = [(noise, sketch) for noise in NOISES for sketch in SKETCHES]
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 7.2), sharex=True, sharey="col")
    for row, (noise, sketch) in enumerate(rows):
        for column, (gamma, rho) in enumerate(RATIOS):
            ax = axes[row, column]
            subset = summary[(summary.gamma == gamma) & (summary.rho == rho) &
                             (summary.noise == noise) & (summary.sketch == sketch)]
            for method in PLOT_METHODS:
                values = subset[subset.method == method].sort_values("n")
                ax.errorbar(values.n, values.mae, yerr=1.96 * values.mae_mcse,
                            color=COLORS[method], marker=MARKERS[method],
                            label=LABELS[method], capsize=2, elinewidth=.8,
                            **styles[method])
            ax.set(xscale="log", yscale="log", xticks=SAMPLE_SIZES)
            ax.set_xticklabels([str(n) for n in SAMPLE_SIZES], fontsize=10)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="both", which="both", labelsize=10)
            name = "row" if sketch == "row" else "weighted Haar"
            noise_label = "Gaussian" if noise == "gaussian" else "Rademacher"
            panel = chr(ord("a") + 2 * row + column)
            ax.set_title(f"({panel}) {noise_label} errors, {name}", fontsize=10, pad=6)
            if row == 3:
                ax.set_xlabel(r"Sample size $n$", fontsize=10)
    # Fix limits after populating all shared axes: repeated log-scale setup on
    # initially empty shared axes can otherwise leave a large unused range.
    axes[0, 0].set_xlim(110, 600)
    for column, (gamma, rho) in enumerate(RATIOS):
        subset = summary[(summary.gamma == gamma) & (summary.rho == rho) &
                         summary.method.isin(PLOT_METHODS)]
        low = float((subset.mae - 1.96 * subset.mae_mcse).min())
        high = float((subset.mae + 1.96 * subset.mae_mcse).max())
        axes[0, column].set_ylim(0.87 * low, 1.14 * high)
        ticks = [0.15, 0.2, 0.3, 0.4] if column == 0 else [0.3, 0.5, 1.0, 1.5]
        axes[0, column].yaxis.set_major_locator(FixedLocator(ticks))
        axes[0, column].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes[0, column].yaxis.set_minor_locator(NullLocator())
        fig.text(0.315 if column == 0 else 0.765, 0.975,
                 rf"$(d/n,m/n)=({gamma:.2f},{rho:.2f})$",
                 ha="center", va="top", fontsize=11)
    fig.text(0.012, 0.55, "Mean absolute error", ha="left", va="center",
             rotation="vertical", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.53, 0.01), frameon=False,
               columnspacing=1.6, handletextpad=0.5, handlelength=2.7)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.18,
                        wspace=0.30, hspace=0.62)
    folder = Path(out) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"accuracy.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_table(summary, out):
    """LaTeX table fragment: MAE (Monte Carlo SE), one common sample size."""
    subset = summary[summary.n == max(SAMPLE_SIZES)]
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Sketch & Errors & $(d/n,m/n)$ & $\hR_{\mathrm{obs}}$ & "
             r"$\hR_{\mathrm{sens}}$ & $\hR_{\GCV}$ & $\widehat R_{\perp,\bS}$ \\",
             r"\midrule"]
    for ratio_index, (gamma, rho) in enumerate(RATIOS):
        if ratio_index:
            lines.append(r"\midrule")
        for sketch in SKETCHES:
            for noise in NOISES:
                group = subset[(subset.gamma == gamma) & (subset.rho == rho) &
                               (subset.sketch == sketch) & (subset.noise == noise)]
                entries = []
                for method in METHODS:
                    values = group[group.method == method]
                    entries.append("---" if values.empty else
                                   f"{values.mae.iloc[0]:.3f} ({values.mae_mcse.iloc[0]:.3f})")
                name = "Row" if sketch == "row" else "Weighted Haar"
                noise_name = "Gaussian" if noise == "gaussian" else "Rademacher"
                lines.append(f"{name} & {noise_name} & $({gamma:.2f},{rho:.2f})$ & " +
                             " & ".join(entries) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (Path(out) / "accuracy_table.tex").write_text("\n".join(lines))


def write_report(frame, summary, paired, out, args):
    failures = int((frame.status != "ok").sum())
    corrected_vs_gcv = paired[(paired.first_method == "corrected") &
                             (paired.second_method == "gcv")]
    corrected_vs_sens = paired[(paired.first_method == "corrected") &
                              (paired.second_method == "population_sensitivity_corrected")]
    lines = ["# Accuracy of corrected risk estimation", "",
        f"Seed: {args.seed}. Independent repetitions per setting: {args.reps:,}.", "",
        "The 24 settings cross n = 128, 256, 512; (d/n,m/n) = (0.25,0.50), "
        "(0.50,0.75); row and weighted Haar sketches; and Gaussian and Rademacher errors. "
        "The Gaussian design has AR(1) covariance with correlation 0.5. Noise variance is one. "
        "The coefficient vector is fixed and sparse, with beta' Sigma beta = 4. Weighted Haar "
        "sketches have alternating row weights 1 and 2 and trace(S' S) = n.", "",
        "Every method targets the exact model risk 1 + (beta_hat-beta)' Sigma "
        "(beta_hat-beta) of its own fitted model. Methods and sketches share each data draw; "
        "Gaussian and Rademacher errors share the feature draw and sketch. Replicates are "
        "independent; changing the number of workers does not change their random streams. "
        "Where dimensions coincide, the Gaussian row-sketch draws are also shared with the "
        "threshold study. That overlap is a paired reuse of data, not an independent replication.", "",
        "The corrected estimate uses d/b_A; the population-sensitivity corrected estimate "
        "uses the known population covariance. GCV uses the full residual and d degrees of "
        "freedom. Complementary risk estimates are summarized only under Gaussian errors. "
        "They remain in the raw file for both error distributions, but no exact Gaussian "
        "calibration claim is made for the Rademacher values.", "",
        f"Attempted fits: {len(frame):,}; numerical failures: {failures:,}. "
        "Failures are retained with their draw IDs and reason; no draw is replaced and no "
        "successful value is trimmed. Each metric lists attempted, available and unavailable counts.", "",
        "## Paired comparisons", "",
        "A negative paired MAE difference favors the first method. Intervals are the mean "
        "paired absolute-error difference plus or minus 1.96 Monte Carlo standard errors. "
        "They are pointwise intervals over independent replicates, without simultaneous coverage.", ""]
    for name, comparisons in (("GCV", corrected_vs_gcv),
                              ("population-sensitivity corrected estimate", corrected_vs_sens)):
        better = int((comparisons.ci_high < 0).sum())
        worse = int((comparisons.ci_low > 0).sum())
        lines.append(f"Against {name}, the corrected estimate has a paired 95% interval "
                     f"below zero in {better}/{len(comparisons)} settings, above zero in "
                     f"{worse}/{len(comparisons)}, and overlapping zero in "
                     f"{len(comparisons) - better - worse}/{len(comparisons)}.")
        lines.append("")
    lines += ["## Files", "",
        "- `raw.csv.gz`: all attempted fits, exact model risks, estimates, errors, and draw IDs.",
        "- `summary.csv`: bias, MAE, MSE, RMSE, Monte Carlo standard errors, and sqrt(n) MAE.",
        "- `paired_comparisons.csv`: all within-fit paired MAE comparisons and 95% intervals.",
        "- `accuracy_table.tex`: n = 512 MAE (Monte Carlo standard error) table.",
        "- `figures/accuracy.pdf` and `.png`: MAE versus n; bars are pointwise 95% Monte Carlo intervals.",
        "- `failures.csv`: all failures, including draw IDs (empty if there were none).",
        "- `metadata.json`: settings, counts, random seed, and provenance.", ""]
    (Path(out) / "short_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.reps < 1 or args.jobs < 1:
        parser.error("--reps and --jobs must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.csv.gz").exists() or (out / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {out}")
    start = time.perf_counter()
    tasks = [(cell, rep) for cell in cells() for rep in range(args.reps)]
    results = run_parallel(partial(one_rep, seed=args.seed), tasks, args.jobs)
    frame = pd.DataFrame([row for result in results for row in result])
    frame = frame.sort_values(GROUP_KEYS + ["rep"]).reset_index(drop=True)
    temporary = out / ".raw.csv.gz.tmp"
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(out / "raw.csv.gz")
    frame[frame.status != "ok"].to_csv(out / "failures.csv", index=False)
    counts = dict(settings=24, repetitions_per_setting=args.reps, fits_attempted=len(frame),
                  fits_successful=int((frame.status == "ok").sum()),
                  numerical_failures=int((frame.status != "ok").sum()),
                  independent_tasks=len(tasks),
                  complete_for_reporting=bool((frame.status == "ok").all()))
    if counts["numerical_failures"]:
        save_metadata(out, "accuracy", args, counts, time.perf_counter() - start)
        raise RuntimeError(
            f"Accuracy study has {counts['numerical_failures']} numerical failures. "
            f"Raw draws, failure records, and metadata are preserved in {out}. "
            "No draws were replaced; no partial-cell results were published.")
    summary, paired = summaries(frame)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_comparisons.csv", index=False)
    write_table(summary, out)
    make_figure(summary, out)
    write_report(frame, summary, paired, out, args)
    counts.update(summary_rows=len(summary), paired_comparison_rows=len(paired))
    save_metadata(out, "accuracy", args, counts, time.perf_counter() - start)
    print(f"Accuracy study complete: {len(frame):,} attempted fits; output: {out}", flush=True)


if __name__ == "__main__":
    main()
