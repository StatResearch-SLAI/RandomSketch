#!/usr/bin/env python3
"""Supplementary isotropic minimum-norm risk benchmarks from original data.

Each replicate draws an entire Gaussian linear-model dataset and independently
draws row, Haar, and raw Gaussian sketches. The three fits share that dataset.
No scalar risk law, chi-square representation, or limiting risk is a generator.
"""

from functools import lru_cache
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from common import (DEFAULT_REPS, NumericalFailure, fit_compressed, make_sketch,
                    run_parallel, save_metadata, seed_rng, setup_plotting)


STUDY = "isotropic_minnorm_benchmarks"
DEFAULT_ISOTROPIC_SEED = 2026092701
SAMPLE_SIZES = (128, 256, 512)
RATIOS = ((0.75, 0.50), (1.25, 0.50))
FAMILIES = ("row", "haar", "gaussian")
FAMILY_LABELS = {"row": "Row", "haar": "Haar", "gaussian": "Dense Gaussian"}
COLORS = {"row": "#777777", "haar": "#009E73", "gaussian": "#0072B2"}
GROUP_KEYS = ["n", "d", "m", "gamma", "rho", "family"]
NUMERICAL_ERRORS = (NumericalFailure, np.linalg.LinAlgError, FloatingPointError,
                    OverflowError)


def cells():
    return [dict(n=n, d=round(n*gamma), m=round(n*rho), gamma=gamma, rho=rho)
            for n in SAMPLE_SIZES for gamma, rho in RATIOS]


@lru_cache(maxsize=12)
def signal(d):
    beta = np.zeros(d)
    support = min(10, d)
    beta[:support] = 2/np.sqrt(support)
    beta.setflags(write=False)
    return beta


def benchmarks(n, d, m):
    if not (m < n and d > m+1):
        raise ValueError("The benchmark requires m<n and d>m+1")
    gamma, rho = d/n, m/n
    exact_mean = 1 + 4*(1-m/d) + m/(d-m-1)
    risk_limit = 4*(1-rho/gamma) + gamma/(gamma-rho)
    return dict(risk_mean_exact=exact_mean, complementary_mean_exact=exact_mean,
                gcv_mean_exact=exact_mean/(1-rho), risk_limit=risk_limit,
                complementary_limit=risk_limit, gcv_limit=risk_limit/(1-rho))


def base_record(cell, rep, seed, family):
    n, d, m = (cell[key] for key in ("n", "d", "m"))
    return dict(cell, study=STUDY, family=family, rep=rep, seed=seed,
                covariance="identity", noise="gaussian", sigma2=1.0,
                signal_norm_squared=4.0,
                data_draw_id=f"{seed}:{STUDY}:n{n}:d{d}:rep{rep}",
                sketch_draw_id=f"{seed}:{STUDY}:n{n}:d{d}:m{m}:{family}:rep{rep}",
                status="numerical_failure", failure_type="", failure_message="",
                risk=np.nan, complementary=np.nan, gcv=np.nan,
                complementary_error=np.nan, gcv_error=np.nan,
                complementary_absolute_error=np.nan, gcv_absolute_error=np.nan,
                risk_limit_squared_error=np.nan, complementary_limit_squared_error=np.nan,
                full_residual_sse=np.nan, interpolation_relative_error=np.nan,
                residual_sketch_relative_error=np.nan,
                diagnostic_svd=(rep == 0), compressed_condition=np.nan,
                **benchmarks(n, d, m))


def simulate_replicate(item):
    """Fit three actual compressed regressions to one shared original dataset."""
    cell, rep, seed = item
    n, d, m = (cell[key] for key in ("n", "d", "m"))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            beta = signal(d)
            x = seed_rng(seed, STUDY, "features", n, d, rep).standard_normal((n, d))
            eps = seed_rng(seed, STUDY, "errors", n, d, rep).standard_normal(n)
            y = x @ beta + eps
    except NUMERICAL_ERRORS as exc:
        records = [base_record(cell, rep, seed, family) for family in FAMILIES]
        for record in records:
            record.update(failure_type=type(exc).__name__, failure_message=str(exc))
        return records

    records = []
    for family in FAMILIES:
        record = base_record(cell, rep, seed, family)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                rng = seed_rng(seed, STUDY, "sketch", n, d, m, family, rep)
                sketch = make_sketch(n, m, family, rng)
                compressed_x, compressed_y = sketch.apply(x), sketch.apply(y)
                # The QR solve uses the raw compressed equations and their
                # minimum Euclidean-norm solution. The optional SVD only
                # diagnoses rank on the prespecified first replicate.
                fit = fit_compressed(compressed_x, compressed_y, diagnostics=(rep == 0))
                beta_hat = fit["beta_hat"]
                residual = y-x @ beta_hat
                delta = beta_hat-beta
                risk = float(1+delta @ delta)
                sse = float(residual @ residual)
                complementary = sse/(n-m)
                gcv = (sse/n)/(1-m/n)**2
                relative_interpolation = float(np.linalg.norm(compressed_x @ beta_hat-compressed_y)
                                               / max(1., np.linalg.norm(compressed_y)))
                relative_residual_sketch = float(np.linalg.norm(sketch.apply(residual))
                                                / max(1., np.linalg.norm(compressed_y)))
                values = (risk, sse, complementary, gcv,
                          relative_interpolation, relative_residual_sketch)
                if not all(np.isfinite(value) and value >= 0 for value in values):
                    raise NumericalFailure("Nonfinite or negative risk, estimate, or residual diagnostic")
                if max(relative_interpolation, relative_residual_sketch) > 1e-8:
                    raise NumericalFailure("Compressed interpolation residual exceeds relative tolerance 1e-8")
                complementary_error, gcv_error = complementary-risk, gcv-risk
                risk_limit_squared_error = float(np.square(risk-record["risk_limit"]))
                complementary_limit_squared_error = float(np.square(complementary-record["risk_limit"]))
                record.update(
                    status="ok", risk=risk, complementary=complementary, gcv=gcv,
                    complementary_error=complementary_error, gcv_error=gcv_error,
                    complementary_absolute_error=abs(complementary_error),
                    gcv_absolute_error=abs(gcv_error),
                    risk_limit_squared_error=risk_limit_squared_error,
                    complementary_limit_squared_error=complementary_limit_squared_error,
                    full_residual_sse=sse,
                    interpolation_relative_error=relative_interpolation,
                    residual_sketch_relative_error=relative_residual_sketch,
                    compressed_condition=float(fit["condition"]))
        except NUMERICAL_ERRORS as exc:
            record.update(failure_type=type(exc).__name__, failure_message=str(exc))
        records.append(record)
    return records


def mean_and_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values)))


def add_summary(row, name, mean, mcse):
    row.update({name: mean, name+"_mcse": mcse,
                name+"_ci_low": mean-1.96*mcse, name+"_ci_high": mean+1.96*mcse})


def summarize(raw, repetitions):
    """Keep estimation errors and concentration around a deterministic limit distinct."""
    rows = []
    for key, group in raw.groupby(GROUP_KEYS, sort=True):
        row = dict(zip(GROUP_KEYS, key))
        failed = int(group.status.ne("ok").sum())
        row.update(repetitions_requested=repetitions, repetitions_attempted=len(group),
                   repetitions_completed=len(group)-failed, repetitions_failed=failed,
                   status="ok" if failed == 0 and len(group) == repetitions else "incomplete",
                   **benchmarks(int(row["n"]), int(row["d"]), int(row["m"])))
        names = [quantity+"_mean" for quantity in ("risk", "complementary", "gcv")]
        names += [quantity+metric for quantity in ("complementary", "gcv")
                  for metric in ("_bias", "_mae")]
        names += [quantity+"_limit_rmse" for quantity in ("risk", "complementary")]
        for name in names:
            add_summary(row, name, np.nan, np.nan)
        row["max_interpolation_relative_error"] = np.nan
        row["max_residual_sketch_relative_error"] = np.nan
        if row["status"] == "ok":
            for quantity in ("risk", "complementary", "gcv"):
                add_summary(row, quantity+"_mean", *mean_and_mcse(group[quantity]))
            for quantity in ("complementary", "gcv"):
                add_summary(row, quantity+"_bias", *mean_and_mcse(group[quantity+"_error"]))
                add_summary(row, quantity+"_mae", *mean_and_mcse(group[quantity+"_absolute_error"]))
            for quantity in ("risk", "complementary"):
                mse, mse_mcse = mean_and_mcse(group[quantity+"_limit_squared_error"])
                rmse = float(np.sqrt(mse))
                rmse_mcse = mse_mcse/(2*rmse) if rmse > 0 else np.nan
                add_summary(row, quantity+"_limit_rmse", rmse, rmse_mcse)
            row["max_interpolation_relative_error"] = float(group.interpolation_relative_error.max())
            row["max_residual_sketch_relative_error"] = float(group.residual_sketch_relative_error.max())
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(summary, output):
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    # Family patterns remain visible between overlapping markers and curves.
    family_styles = {
        "row": dict(linestyle="-", marker="o", markersize=5.8, linewidth=1.5),
        "haar": dict(linestyle=(0, (5, 2.5)), marker="s", markersize=4.8, linewidth=1.3),
        "gaussian": dict(linestyle=(0, (1, 1.8)), marker="^", markersize=4.6, linewidth=1.3),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.5), sharex="col")
    for column, (gamma, rho) in enumerate(RATIOS):
        subset = summary[(summary.gamma == gamma) & (summary.rho == rho)]
        for family in FAMILIES:
            values = subset[(subset.family == family) & (subset.status == "ok")].sort_values("n")
            for quantity in ("risk", "complementary"):
                style = dict(family_styles[family])
                style["markerfacecolor"] = COLORS[family] if quantity == "risk" else "none"
                if quantity == "complementary":
                    style["markersize"] += 1.2
                for row_index, metric in ((0, "_mean"), (1, "_limit_rmse")):
                    field = quantity+metric
                    axes[row_index, column].errorbar(
                        values.n, values[field], yerr=1.96*values[field+"_mcse"],
                        color=COLORS[family], markeredgewidth=1.0,
                        capsize=2, elinewidth=.8, **style)
        theory = subset[subset.family == "row"].sort_values("n")
        axes[0, column].plot(theory.n, theory.risk_mean_exact, color="black",
                             linestyle="-.", linewidth=1.2)
        axes[0, column].axhline(float(theory.risk_limit.iloc[0]), color="#999999",
                                linestyle=":", linewidth=1.5)
        axes[0, column].set_title(rf"$(\gamma,\rho)=({gamma:.2f},{rho:.2f})$", fontsize=10, pad=19)
        axes[1, column].set_xlabel(r"Sample size $n$")
        for row_index in (0, 1):
            ax = axes[row_index, column]
            ax.set_xscale("log")
            ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(108, 605)
            ax.grid(axis="y", alpha=.18)
            ax.text(0, 1.025, f"({chr(ord('a')+2*row_index+column)})",
                    transform=ax.transAxes, fontsize=10)
    axes[0, 0].set_ylabel("Mean risk or risk estimate")
    axes[1, 0].set_ylabel("RMSE around limiting risk")
    family_handles = [Line2D([], [], color=COLORS[family], markerfacecolor="none",
                             label=FAMILY_LABELS[family], **family_styles[family])
                      for family in FAMILIES]
    quantity_handles = [Line2D([], [], color="black", linestyle="none", marker="o",
                              markersize=4.5, label=r"Risk $R_{\min,\mathbf{S}}$"),
                        Line2D([], [], color="black", linestyle="none", marker="o",
                              markerfacecolor="none", markersize=5.7,
                              label=r"Complementary $\widehat R_{\min,\mathbf{S}}$"),
                        Line2D([], [], color="black", linestyle="-.", label="Exact mean"),
                        Line2D([], [], color="#999999", linestyle=":", label="Limiting risk")]
    fig.legend(handles=family_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.52, .085), fontsize=9, handlelength=3)
    fig.legend(handles=quantity_handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.52, -.002), fontsize=9, columnspacing=1.8, handlelength=3)
    fig.subplots_adjust(left=.12, right=.98, top=.86, bottom=.25, hspace=.50, wspace=.30)
    folder = Path(output)/"figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder/f"isotropic_benchmarks.{suffix}", bbox_inches="tight", dpi=250)
    plt.close(fig)


def write_report(raw, summary, args):
    failures = int(raw.status.ne("ok").sum())
    lines = ["# Supplementary isotropic minimum-norm benchmarks", "",
             f"Master seed: {args.seed}. Repetitions per setting: {args.reps:,}. "
             f"Attempted fits: {len(raw):,}; numerical failures: {failures:,}.", "",
             "The 18 settings cross n=128,256,512; (d/n,m/n)=(0.75,0.50),(1.25,0.50); "
             "and row, Haar, and raw dense Gaussian sketches. Features and errors are independent "
             "standard Gaussian variables. The covariance is deliberately the identity, as required "
             "by the isotropic minimum-norm benchmark theorem. The fixed coefficient vector has "
             "its first min(10,d) coordinates equal and squared Euclidean norm four.", "",
             "Each repetition generates an original X and y=X beta+epsilon. The three sketch "
             "families share these data and use independently keyed sketch streams, independent "
             "of the data. Row and Haar sketches have orthogonal rows of length sqrt(n/m); "
             "the dense Gaussian sketch has independent N(0,1/m) entries. The raw compressed "
             "equations are solved by a minimum-norm QR solve, with no ridge or whitening. "
             "The Gaussian law of a scalar risk or pivot is never used to generate observations.", "",
             "Exact response risk is R=1+||beta_hat-beta||^2. All fits interpolate the compressed "
             "equations, so the complementary risk estimate is ||y-X beta_hat||^2/(n-m). "
             "Conventional GCV is [||y-X beta_hat||^2/n]/(1-m/n)^2. Its exact multiplicative "
             "inflation relative to the complementary estimate is 1/(1-m/n).", "",
             "The exact mean risk is 1+4(1-m/d)+m/(d-m-1). The limiting risk is "
             "4(1-rho/gamma)+gamma/(gamma-rho), where gamma=d/n and rho=m/n. "
             "The complementary estimate has the same exact mean and limit. GCV has exact "
             "mean and limiting value larger by 1/(1-rho); these quantities are archived, "
             "but GCV is omitted from the figure to keep the other curves readable.", "",
             "The upper panels compare empirical means with the exact finite-sample mean "
             "and proportional limit. The lower panels report RMSE of the realized risk "
             "and complementary estimate around that deterministic limit. These are "
             "concentration errors, distinct from estimating the risk of an individual fitted "
             "model. The archive separately records each risk estimate's bias and MAE relative "
             "to its own fit's realized risk.", "",
             "Monte Carlo standard errors for means, biases, and MAEs are sample standard "
             "deviations (denominator repetitions-1) divided by sqrt(repetitions). For a "
             "concentration RMSE, the delta-method standard error is the standard error of "
             "the mean squared deviation divided by twice the RMSE. Bars and archived "
             "intervals use 1.96 standard errors and are pointwise normal approximations. "
             "The three families are paired through their data; results across families are "
             "not independent replications.", ""]
    if failures == 0:
        maximum = max(float(raw.interpolation_relative_error.max()),
                      float(raw.residual_sketch_relative_error.max()))
        lines += [f"All {len(raw):,} fits completed without replacements or trimming. "
                  f"The largest relative compressed-interpolation diagnostic is {maximum:.3e}. "
                  "An SVD rank diagnostic was also performed on the prespecified first "
                  "repetition in every setting, without changing the QR estimator.", "",
                  "| n | d/n | Sketch | Mean R (MCSE) | Mean complementary (MCSE) | "
                  "Mean GCV (MCSE) | Exact mean R | Limiting R |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for _, row in summary.sort_values(["gamma", "n", "family"]).iterrows():
            lines.append(f"| {int(row.n)} | {row.gamma:.2f} | {FAMILY_LABELS[row.family]} | "
                         f"{row.risk_mean:.4f} ({row.risk_mean_mcse:.4f}) | "
                         f"{row.complementary_mean:.4f} ({row.complementary_mean_mcse:.4f}) | "
                         f"{row.gcv_mean:.4f} ({row.gcv_mean_mcse:.4f}) | "
                         f"{row.risk_mean_exact:.4f} | {row.risk_limit:.4f} |")
        lines += ["", "![Isotropic minimum-norm benchmarks](figures/isotropic_benchmarks.png)", ""]
    else:
        lines += ["The run is incomplete. Failed draws are retained without replacement; "
                  "affected settings have no inferential summaries.", ""]
    lines += ["Files: [raw draws](raw.csv.gz), [summaries](summary.csv), "
              "[failure records](failures.csv), and [metadata](metadata.json).", ""]
    (args.output/"short_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_ISOTROPIC_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.reps < 2 or args.jobs < 1:
        parser.error("Require at least two repetitions and one worker")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/name).exists() for name in ("raw.csv.gz", "metadata.json", "summary.csv")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    start = time.perf_counter()
    tasks = [(cell, rep, args.seed) for cell in cells() for rep in range(args.reps)]
    blocks = run_parallel(simulate_replicate, tasks, args.jobs)
    raw = pd.DataFrame([record for records in blocks for record in records])
    raw = raw.sort_values(["n", "d", "m", "family", "rep"]).reset_index(drop=True)
    summary = summarize(raw, args.reps)
    raw.to_csv(args.output/"raw.csv.gz", index=False, compression="gzip")
    raw.loc[raw.status.ne("ok")].to_csv(args.output/"failures.csv", index=False)
    summary.to_csv(args.output/"summary.csv", index=False)
    failed = int(raw.status.ne("ok").sum())
    counts = dict(settings=len(cells())*len(FAMILIES), independent_datasets=len(tasks),
                  attempted_fits=len(raw), completed_fits=len(raw)-failed, failed_fits=failed,
                  repetitions_per_setting=args.reps,
                  successful_svd_diagnostics=int((raw.diagnostic_svd & raw.status.eq("ok")).sum()),
                  pairing="Three sketch families share each original X and y; sketches use independent streams",
                  scope="Isotropic minimum-norm benchmark and concentration, separate from estimation error",
                  catalog=cells(), sketch_families=FAMILIES)
    population_info = dict(design="Gaussian", covariance="identity", noise="Gaussian",
                           noise_variance=1.0, prediction_signal=4.0, signal_norm_squared=4.0,
                           signal_direction="first min(10,d) coordinates equal; others zero")
    save_metadata(args.output, STUDY, args, counts, time.perf_counter()-start,
                  population_info=population_info,
                  seed_scheme="SHA256 keyed isotropic data streams shared across three families; independent sketch streams")
    write_report(raw, summary, args)
    if failed:
        raise RuntimeError(f"Preserved {failed} failed fits; no replacements or trimming")
    make_figure(summary, args.output)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(raw):,} fits from {len(tasks):,} independent datasets to {args.output}", flush=True)


if __name__ == "__main__":
    main()
