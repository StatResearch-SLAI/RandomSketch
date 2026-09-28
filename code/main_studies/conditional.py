#!/usr/bin/env python3
"""Main study 1: conditional accuracy of the complementary risk estimate.

Freeze an actual raw sketch and its Gaussian row-space data. Draw all d feature
coordinates and the noise in each complementary direction for each replicate.
The chi-square distribution is used only to form the exact risk interval; no
chi-square or risk-ratio samples are used to generate the observations.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2, t

try:
    from .common import (DEFAULT_REPS, DEFAULT_SEED, NumericalFailure,
                         fit_compressed, make_sketch, population, run_parallel,
                         save_metadata, seed_rng, setup_plotting)
except ImportError:
    from common import (DEFAULT_REPS, DEFAULT_SEED, NumericalFailure,
                        fit_compressed, make_sketch, population, run_parallel,
                        save_metadata, seed_rng, setup_plotting)


STUDY = "complementary_calibration"
FAMILIES = ("row", "gaussian")
FAMILY_LABELS = {"row": "Row", "gaussian": "Dense Gaussian"}
COLORS = {"row": "#777777", "gaussian": "#0072B2"}
MARKERS = {"row": "o", "gaussian": "s"}
SETTING_KEYS = ["setting_id", "n", "m", "d", "family", "complement_df", "regime"]
METRICS = ("mean_ratio", "variance_ratio", "coverage")
AUDITS = ("orthogonality_error", "summary_error", "refit_error",
          "complement_residual_error", "full_residual_error")


def settings_catalog():
    """The six prespecified settings, unchanged by CLI repetition controls."""
    return [dict(setting_id=f"d{d:03d}-{family}", n=256, m=128, d=d,
                 family=family, complement_df=128,
                 regime="d < m" if d < 128 else "d > m")
            for d in (64, 192, 320) for family in FAMILIES]


def freeze_summary(setting, seed, summary_index):
    """Generate S, QX and Qepsilon, then fit the literal compressed equations.

    Q=(SS')^{-1/2}S has orthonormal rows. Given S, QX and Qepsilon
    are independent standard Gaussian rows with the specified covariance.
    For d<m, the factor S Q' is retained in the least-squares objective.
    """
    n, m, d = (setting[key] for key in ("n", "m", "d"))
    key = (STUDY, setting["setting_id"], summary_index)
    sigma, chol, beta = population(d)
    sketch = make_sketch(n, m, setting["family"],
                         seed_rng(seed, *key, "sketch"))
    s = sketch.to_dense()
    left, singular, vt = np.linalg.svd(s, full_matrices=False)
    tolerance = np.finfo(float).eps * max(s.shape) * singular[0]
    if not np.all(np.isfinite(singular)) or singular[-1] <= tolerance:
        raise NumericalFailure("Sketch is not numerically full row rank")
    q = left @ vt
    complete, _ = np.linalg.qr(q.T, mode="complete")
    q_perp = complete[:, m:].T
    row_rng = seed_rng(seed, *key, "row-space-data")
    x_row = row_rng.standard_normal((m, d)) @ chol.T
    eps_row = row_rng.standard_normal(m)
    y_row = x_row @ beta + eps_row
    row_transform = s @ q.T
    compressed_x = row_transform @ x_row
    compressed_y = row_transform @ y_row
    fit = fit_compressed(compressed_x, compressed_y)
    beta_hat = fit["beta_hat"]
    delta = beta_hat - beta
    latent_delta = chol.T @ delta
    risk = float(1.0 + latent_delta @ latent_delta)
    row_residual = y_row - x_row @ beta_hat
    row_residual_sse = float(row_residual @ row_residual)
    orthogonality_error = max(
        float(np.linalg.norm(q @ q.T - np.eye(m), ord="fro") / np.sqrt(m)),
        float(np.linalg.norm(q_perp @ q_perp.T - np.eye(n-m), ord="fro")
              / np.sqrt(n-m)),
        float(np.linalg.norm(q @ q_perp.T, ord="fro") / np.sqrt(m)))
    if orthogonality_error > 1e-10 or not np.isfinite(risk) or risk <= 0:
        raise NumericalFailure("Invalid orthogonal decomposition or model risk")
    return dict(S=s, Q=q, Q_perp=q_perp, X_row=x_row, eps_row=eps_row,
                y_row=y_row, compressed_x=compressed_x,
                compressed_y=compressed_y, Sigma=sigma, chol=chol,
                beta=beta, beta_hat=beta_hat, latent_delta=latent_delta,
                risk=risk, row_residual=row_residual,
                row_residual_sse=row_residual_sse,
                orthogonality_error=orthogonality_error,
                compressed_condition=float(fit["condition"]), key=key)


def reconstruction_audit(frozen, latent_complement, noise_complement, residual):
    """Reconstruct one full dataset without an interpolation assumption."""
    x_perp = latent_complement @ frozen["chol"].T
    x = frozen["Q"].T @ frozen["X_row"] + frozen["Q_perp"].T @ x_perp
    eps = frozen["Q"].T @ frozen["eps_row"] + frozen["Q_perp"].T @ noise_complement
    y = x @ frozen["beta"] + eps
    b, sy = frozen["S"] @ x, frozen["S"] @ y
    refit = fit_compressed(b, sy)["beta_hat"]
    full_residual = y - x @ refit
    reconstructed = (frozen["Q"].T @ frozen["row_residual"]
                     + frozen["Q_perp"].T @ residual)
    audit = dict(
        orthogonality_error=frozen["orthogonality_error"],
        summary_error=max(
            np.linalg.norm(b-frozen["compressed_x"])
            / max(1., np.linalg.norm(frozen["compressed_x"])),
            np.linalg.norm(sy-frozen["compressed_y"])
            / max(1., np.linalg.norm(frozen["compressed_y"]))),
        refit_error=np.linalg.norm(refit-frozen["beta_hat"])
        / max(1., np.linalg.norm(frozen["beta_hat"])),
        complement_residual_error=np.linalg.norm(frozen["Q_perp"] @ full_residual-residual)
        / max(1., np.linalg.norm(residual)),
        full_residual_error=np.linalg.norm(full_residual-reconstructed)
        / max(1., np.linalg.norm(full_residual)))
    if not all(np.isfinite(value) and value <= 1e-8 for value in audit.values()):
        raise NumericalFailure(f"Full-data reconstruction failed: {audit}")
    return {key: float(value) for key, value in audit.items()}


def one_summary(item):
    setting, seed, summary_index, reps, output = item
    summary_id = f'{setting["setting_id"]}-summary-{summary_index:03d}'
    base = dict(setting, summary_id=summary_id, summary_index=summary_index, seed=seed)
    records = []
    diagnostics = dict(base, status="failed", error="")
    frozen_dir = Path(output) / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            frozen = freeze_summary(setting, seed, summary_index)
            artifact = frozen_dir / f"{summary_id}.npz"
            artifact_keys = ("S", "Q", "Q_perp", "X_row", "eps_row", "y_row",
                             "compressed_x", "compressed_y", "Sigma", "beta",
                             "beta_hat", "row_residual", "risk")
            np.savez_compressed(artifact, **{key: frozen[key] for key in artifact_keys})
            artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            risk, df = frozen["risk"], setting["complement_df"]
            diagnostics.update(
                risk=risk, row_residual_sse=frozen["row_residual_sse"],
                compressed_condition=frozen["compressed_condition"],
                prediction_signal=float(frozen["beta"] @ frozen["Sigma"] @ frozen["beta"]),
                frozen_file=str(artifact.relative_to(output)), frozen_sha256=artifact_hash)
            quantile_low, quantile_high = chi2.ppf([.025, .975], df)
            audit = {key: np.nan for key in AUDITS}
            # Fixed-size keyed batches give the same prefix if --reps changes.
            # Each draw includes df*d independent Gaussian feature coordinates.
            for start in range(0, reps, 25):
                rng = seed_rng(seed, *frozen["key"], "complement-data", start // 25)
                latent = rng.standard_normal((25, df, setting["d"]))
                noise = rng.standard_normal((25, df))
                # Literal y_perp - X_perp beta_hat, evaluated through the
                # covariance factor to avoid redundant dense matrix products.
                residuals = noise - latent @ frozen["latent_delta"]
                if start == 0:
                    audit = reconstruction_audit(frozen, latent[0], noise[0], residuals[0])
                for j in range(min(25, reps-start)):
                    rep = start+j
                    complementary_sse = float(residuals[j] @ residuals[j])
                    estimate = complementary_sse / df
                    ratio = estimate / risk
                    ci_lower = complementary_sse / quantile_high
                    ci_upper = complementary_sse / quantile_low
                    if not all(np.isfinite(value) for value in
                               (complementary_sse, estimate, ratio, ci_lower, ci_upper)):
                        raise NumericalFailure(f"Nonfinite output for draw {rep}")
                    records.append(dict(
                        base, rep=rep, status="ok", error="", risk=risk,
                        complementary_sse=complementary_sse,
                        row_residual_sse=frozen["row_residual_sse"],
                        full_residual_sse=complementary_sse+frozen["row_residual_sse"],
                        complementary_risk_estimate=estimate, ratio=ratio,
                        ci_lower=ci_lower, ci_upper=ci_upper,
                        coverage=bool(ci_lower <= risk <= ci_upper)))
            diagnostics.update(status="ok", **audit)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        message = f"{type(error).__name__}: {error}"
        diagnostics.update(status="failed", error=message)
        # Retain completed draws and mark every remaining planned draw failed.
        # An incomplete cluster is never silently used for reported inference.
        records.extend(dict(base, rep=rep, status="failed", error=message)
                       for rep in range(len(records), reps))
    return records, diagnostics


def summarize_frozen(frame, diagnostics):
    """Compute conditional summaries only for fully observed frozen clusters."""
    rows = []
    by_id = diagnostics.set_index("summary_id")
    for summary_id, group in frame.groupby("summary_id", sort=True):
        row = {key: group.iloc[0][key] for key in SETTING_KEYS}
        row.update(summary_id=summary_id,
                   summary_index=int(group.iloc[0].summary_index),
                   attempted_draws=len(group), successful_draws=int((group.status == "ok").sum()),
                   failed_draws=int((group.status != "ok").sum()))
        row["complete"] = row["failed_draws"] == 0 and by_id.loc[summary_id, "status"] == "ok"
        for metric in (*METRICS, "mean_ratio_mcse", "relative_mse", "risk"):
            row[metric] = np.nan
        if row["complete"]:
            ratios = group.ratio.to_numpy(dtype=float)
            df = int(row["complement_df"])
            row.update(risk=float(group.risk.iloc[0]), mean_ratio=float(ratios.mean()),
                       mean_ratio_mcse=float(ratios.std(ddof=1)/np.sqrt(len(ratios))),
                       variance_ratio=float(ratios.var(ddof=1)/(2/df)),
                       coverage=float(group.coverage.to_numpy(dtype=float).mean()),
                       relative_mse=float(np.mean((ratios-1)**2)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_settings(frozen_summary):
    """Student t intervals across independent frozen summaries (default df=19)."""
    rows = []
    for key, group in frozen_summary.groupby(SETTING_KEYS, sort=True):
        row = dict(zip(SETTING_KEYS, key))
        row.update(attempted_summaries=len(group), complete_summaries=int(group.complete.sum()),
                   attempted_draws=int(group.attempted_draws.sum()),
                   failed_draws=int(group.failed_draws.sum()), complete=bool(group.complete.all()),
                   cluster_df=len(group)-1,
                   exact_mean_ratio=1., exact_variance_ratio=1., exact_coverage=.95,
                   exact_normalized_variance=2/row["complement_df"])
        for metric in METRICS:
            for suffix in ("", "_mcse", "_low", "_high"):
                row[metric+suffix] = np.nan
            if row["complete"]:
                values = group[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                se = float(values.std(ddof=1)/np.sqrt(len(values)))
                margin = float(t.ppf(.975, len(values)-1)*se)
                row.update({metric: mean, metric+"_mcse": se,
                            metric+"_low": mean-margin, metric+"_high": mean+margin})
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(settings, output):
    setup_plotting()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.8))
    panels = (("mean_ratio", "(a) Mean / risk", 1.),
              ("variance_ratio", "(b) Variance / exact", 1.),
              ("coverage", "(c) Interval coverage", .95))
    dimensions = (64, 192, 320)
    for ax, (metric, title, target) in zip(axes, panels):
        for family_index, family in enumerate(FAMILIES):
            group = settings[settings.family == family].sort_values("d")
            group = group[group.complete]
            if group.empty:
                ax.plot([], [], MARKERS[family], color=COLORS[family],
                        label=FAMILY_LABELS[family]+" (incomplete)")
                continue
            values = group[metric].to_numpy(dtype=float)
            lo = group[metric+"_low"].to_numpy(dtype=float)
            hi = group[metric+"_high"].to_numpy(dtype=float)
            x = np.array([dimensions.index(int(d)) for d in group.d]) + (family_index-.5)*.16
            ax.errorbar(x, values, yerr=np.vstack((values-lo, hi-values)),
                        fmt=MARKERS[family], color=COLORS[family], capsize=3,
                        markersize=5, linewidth=1.2, label=FAMILY_LABELS[family])
        ax.axhline(target, color="black", linestyle=":", linewidth=1.4,
                   label="Exact benchmark")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(dimensions)), [str(d) for d in dimensions])
        ax.set_xlabel(r"$d$")
        ax.set_xlim(-.4, 2.4)
        bounds = settings[[metric+"_low", metric+"_high"]].to_numpy(dtype=float)
        finite = bounds[np.isfinite(bounds)]
        if finite.size:
            half = max(float(np.max(np.abs(finite-target)))*1.2,
                       .0025 if metric == "coverage" else .004)
            ax.set_ylim(target-half, target+half)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"{STUDY}.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def write_report(frame, frozen_summary, settings, diagnostics, args):
    total_failures = int((frame.status != "ok").sum())
    complete = bool(settings.complete.all())
    lines = ["# Complementary risk estimate: conditional calibration", "",
             f'**Status: {"complete" if complete else "incomplete; see retained failure records"}.**', "",
             f"Six settings use n=256, m=128, d in {{64,192,320}}, and row or raw dense Gaussian sketches. "
             f"Each setting has {args.summaries} independent frozen fitted summaries and "
             f"{args.reps:,} complementary-data draws per summary. "
             f"The run retains {len(frame):,} planned draws, including {total_failures:,} failures.", "",
             "Features have AR(1) covariance Sigma[j,k]=0.5^|j-k|; Gaussian noise has variance one. "
             "The common deterministic sparse coefficient vector is normalized to beta' Sigma beta=4.", "",
             "We freeze S, QX, and Qepsilon, where Q=(SS')^(-1/2)S. For each draw we generate "
             "all independent Gaussian feature coordinates and noise in the n-m complementary directions. "
             "This generates conditional linear-model data, rather than sampling a chi-square law. "
             "The fit solves the raw compressed equations SX and Sy by least squares when d<m and "
             "by the minimum-norm solution when d>m. The raw dense sketch has independent N(0,1/m) entries.", "",
             "The complementary risk estimate is ||Q_perp(y-X beta_hat)||^2/(n-m) in every setting. "
             "For d<m, the full residual also contains the frozen row-space residual; that contribution "
             "is recorded separately and is excluded from this estimate. True response prediction risk is "
             "R=1+(beta_hat-beta)' Sigma (beta_hat-beta). The exact 95% interval for R is "
             "[SSE/chi2_(n-m,.975), SSE/chi2_(n-m,.025)].", "",
             "The figure reports the mean estimate/R, the conditional sample variance of estimate/R "
             "divided by 2/(n-m), and coverage. Targets are 1, 1, and 0.95. First compute each metric "
             "within a frozen summary, then average over summaries. Pointwise 95% Monte Carlo intervals "
             f"use Student t with {args.summaries-1} degrees of freedom across those independent clusters. "
             "The inner draws are not treated as independent training fits. No failed problem is redrawn "
             "or trimmed; any incomplete setting has missing inferential summaries.", "",
             "| Sketch | d | Mean / risk (95% MC interval) | Variance / exact (95% MC interval) | Coverage (95% MC interval) |",
             "|---|---:|---:|---:|---:|"]
    for _, row in settings.sort_values(["d", "family"]).iterrows():
        cells = [f"{row[metric]:.4f} [{row[metric+'_low']:.4f}, {row[metric+'_high']:.4f}]"
                 if row.complete else "Incomplete" for metric in METRICS]
        lines.append(f'| {FAMILY_LABELS[row.family]} | {int(row.d)} | '+" | ".join(cells)+" |")
    good = diagnostics[diagnostics.status == "ok"]
    if not good.empty:
        largest = max(float(good[key].max()) for key in AUDITS)
        lines.extend(["", f"For the first complementary draw of each frozen summary, reconstructing the "
                      f"full data preserves the compressed summaries, refitted coefficients, and complementary "
                      f"and full residuals to a maximum relative discrepancy of {largest:.2e}. "
                      "This audit does not assume interpolation."])
    lines.extend(["", "![Conditional calibration](figures/complementary_calibration.png)", "",
                  "Outputs: [raw draws](raw.csv.gz), [per-frozen-fit summaries](frozen_summaries.csv), "
                  "[setting summaries](settings.csv), [frozen artifacts and diagnostics](frozen_diagnostics.csv), "
                  "[failures](failures.csv), and [provenance](metadata.json). "
                  "Each frozen NPZ stores S, the orthogonal decomposition, compressed data, coefficients, "
                  "and exact risk; its SHA-256 hash is in the diagnostics file.", ""])
    (args.output / "short_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--summaries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.reps < 3 or args.summaries < 2 or args.jobs < 1:
        parser.error("Require at least 3 draws, 2 summaries, and 1 worker")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metadata.json").exists() or (args.output / "raw.csv.gz").exists():
        parser.error("Output already contains a run; choose a fresh directory")
    start = time.perf_counter()
    catalog = settings_catalog()
    tasks = [(setting, args.seed, index, args.reps, str(args.output))
             for setting in catalog for index in range(args.summaries)]
    blocks = run_parallel(one_summary, tasks, args.jobs)
    frame = pd.DataFrame([row for records, _ in blocks for row in records])
    diagnostics = pd.DataFrame([item for _, item in blocks])
    frozen_summary = summarize_frozen(frame, diagnostics)
    settings = summarize_settings(frozen_summary)
    frame.to_csv(args.output / "raw.csv.gz", index=False, compression="gzip")
    frame.loc[frame.status != "ok"].to_csv(args.output / "failures.csv", index=False)
    diagnostics.to_csv(args.output / "frozen_diagnostics.csv", index=False)
    frozen_summary.to_csv(args.output / "frozen_summaries.csv", index=False)
    settings.to_csv(args.output / "settings.csv", index=False)
    make_figure(settings, args.output)
    write_report(frame, frozen_summary, settings, diagnostics, args)
    save_metadata(args.output, STUDY, args,
                  dict(settings=len(catalog), frozen_summaries=len(tasks),
                       complementary_draws=len(frame),
                       successful_draws=int((frame.status == "ok").sum()),
                       failed_draws=int((frame.status != "ok").sum()),
                       reconstruction_audit_refits=int((diagnostics.status == "ok").sum()),
                       conditioning="S, QX, Qepsilon; fresh Gaussian complementary design and noise",
                       uncertainty="Pointwise Student-t intervals across frozen summaries",
                       catalog=catalog), time.perf_counter()-start)
    if not settings.complete.all():
        raise SystemExit("Incomplete simulation: all failures retained; inspect failures.csv")


if __name__ == "__main__":
    main()
