#!/usr/bin/env python3
"""Empirical MSE ordering for weighted Haar and raw dense Gaussian sketches.

The original X,y draws match threshold.py. Five sketch sizes share each dataset
and nested prefixes of one independently generated sketch per family. The
row-sketch scaled orthogonal MSE formulas and crossover lines are not applied here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.stats import norm
from threadpoolctl import threadpool_info, threadpool_limits

from common import (DEFAULT_SEED, NumericalFailure, generate_data,
                    seed_rng, setup_plotting)
from threshold import GAMMAS, MSE_REPS, OFFSETS, SAMPLE_SIZES, sketch_sizes

STUDY = "threshold_shapes"
FAMILIES = ("weighted_haar", "dense_gaussian")
CHECKPOINT_REPS = 100
NUMERIC = ("risk", "complementary", "corrected", "complementary_error",
           "corrected_error", "complementary_squared_error", "corrected_squared_error",
           "b_A", "RSS", "trace_S_transpose_S", "normal_equation_error")
ERRORS = (NumericalFailure, np.linalg.LinAlgError, FloatingPointError)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: digest(folder / name) for name in
            ("threshold_shapes.py", "threshold.py", "common.py")} | {
                "../utils.py": digest(folder.parent / "utils.py")}


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {key: importlib.metadata.version(key) for key in
                         ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl")},
            "threadpools": threadpool_info(), "blas_threads_per_worker": 1}


def specification(reps, seed):
    return {"study": STUDY, "schema": 1, "reps": int(reps), "seed": int(seed),
            "sample_sizes": list(SAMPLE_SIZES), "d_over_n": list(GAMMAS),
            "nominal_offsets": list(OFFSETS), "sketches": list(FAMILIES),
            "grid": [{"n": n, "d": round(n * gamma),
                      "m": list(sketch_sizes(n, round(n * gamma)))}
                     for n in SAMPLE_SIZES for gamma in GAMMAS],
            "rounding": "Nearest integer; exact half-integer ties round upward",
            "noise": "Gaussian N(0,1)", "sigma2": 1.,
            "population": "Gaussian AR(1) correlation 0.5; sparse signal variance 4",
            "pairing": "X,y match threshold.py; five m values share each dataset and nested sketch prefixes",
            "sketch_rules": {"weighted_haar": "Nested Haar rows; alternating weights 1,2; trace(S.T S)=n",
                             "dense_gaussian": "Nested iid standard Gaussian rows divided by sqrt(m); no pathwise normalization"},
            "theory_scope": "Empirical MSE ordering only; scaled orthogonal exact MSE and crossover formulas are not applied"}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _base_record(n, d, m, rep, seed, family):
    return {"study": STUDY, "n": n, "d": d, "m": m, "gamma": d / n,
            "rho": m / n, "noise": "gaussian", "sketch": family, "rep": rep,
            "seed": seed, "design_draw_id": f"{seed}:features:n{n}:d{d}:rep{rep}",
            "data_draw_id": f"{seed}:data:n{n}:d{d}:gaussian:rep{rep}",
            "sketch_draw_id": f"{seed}:threshold-shapes:n{n}:d{d}:{family}:rep{rep}",
            "fit_id": f"{seed}:n{n}:d{d}:rep{rep}:gaussian:{family}:m{m}",
            "covariance": "AR(1), correlation 0.5", "sigma2": 1.,
            "prediction_signal": 4.,
            "nominal_offset": OFFSETS[sketch_sizes(n, d).index(m)],
            "central_setting": m == sketch_sizes(n, d)[len(OFFSETS) // 2],
            "theory_scope": "empirical_non_tight_frame",
            "status": "numerical_failure", "failure_type": "", "failure_message": "",
            **dict.fromkeys(NUMERIC, np.nan)}


def prepare_family(data, sizes, n, d, rep, seed, family):
    """One maximum-size sketch, its row-space basis, and original compression.

    Prefixing an unpivoted Gaussian QR gives Haar rows at every requested size.
    Dense Gaussian prefixes retain literal iid N(0,1/m) entries after scaling.
    """
    maximum = max(sizes)
    rng = seed_rng(seed, "threshold-shapes", family, n, d, rep)
    if family == "weighted_haar":
        basis, factor = qr(rng.standard_normal((n, maximum)), mode="economic",
                           check_finite=False)
        basis *= np.where(np.diag(factor) >= 0., 1., -1.)
        base = basis.T
    elif family == "dense_gaussian":
        base = rng.standard_normal((maximum, n))
        basis, _ = qr(base.T, mode="economic", check_finite=False)
    else:
        raise ValueError(f"Unknown sketch family: {family}")
    if not np.isfinite(base).all() or not np.isfinite(basis).all():
        raise NumericalFailure("Nonfinite sketch or orthogonal row-space basis")
    return {"family": family, "base": base, "basis": basis,
            "BX": base @ data.X, "Cy": base @ data.y}


def fit_family(data, preparation, m):
    """Actual sketched least squares, calibrated observed risk, and projection.

    The orthogonal complement is computed from QR(S.T), avoiding inverse row
    Gram matrices. No simulated risk/residual law replaces the original data.
    """
    n, d = data.X.shape
    base = preparation["base"][:m]
    family = preparation["family"]
    if family == "weighted_haar":
        weights = np.ones(m)
        weights[1::2] = 2.
        scale = weights * np.sqrt(n / (weights @ weights))
    else:
        scale = np.full(m, 1. / np.sqrt(m))
    B = scale[:, None] * preparation["BX"][:m]
    c = scale * preparation["Cy"][:m]
    qfit, r = qr(B, mode="economic", check_finite=False)
    if not np.isfinite(r).all() or np.any(np.abs(np.diag(r)) <= np.finfo(float).tiny):
        raise NumericalFailure("Nonfinite or numerically singular compressed QR")
    estimate = solve_triangular(r, qfit.T @ c, check_finite=False)
    u = data.y - data.X @ estimate
    su = scale * (base @ u)
    au = base.T @ (scale * su)
    trace = float(np.sum(base * base * scale[:, None] ** 2))
    if family == "weighted_haar":
        diagonal = scale ** 2
        b = float(np.sum(diagonal) - np.sum(qfit ** 2 * diagonal[:, None]))
    else:
        sq = base.T @ (scale[:, None] * qfit)
        b = float(trace - np.sum(sq * sq))
    if not np.isfinite(b) or b <= 0.:
        raise NumericalFailure(f"Nonpositive or nonfinite residual sketch mass: {b}")
    rss = float(u @ u)
    corrected_vector = u + d / b * au
    corrected = float(corrected_vector @ corrected_vector / n)
    basis = preparation["basis"][:, :m]
    complement = u - basis @ (basis.T @ u)
    complementary = float(complement @ complement / (n - m))
    error = estimate - data.beta
    risk = float(data.sigma2 + error @ data.Sigma @ error)
    residual = c - B @ estimate
    normal_error = float(np.linalg.norm(B.T @ residual) / max(
        np.linalg.norm(B) * np.linalg.norm(c), np.finfo(float).tiny))
    values = np.array([risk, complementary, corrected, b, rss, trace, normal_error])
    if not np.isfinite(values).all() or np.any(values[:6] < 0.):
        raise NumericalFailure("Nonfinite or negative risk, criterion, or diagnostic")
    if normal_error > 1.e-9:
        raise NumericalFailure("Failed scaled normal-equation diagnostic")
    errors = np.array([complementary - risk, corrected - risk])
    with np.errstate(over="raise", invalid="raise"):
        squares = errors ** 2
    return {"risk": risk, "complementary": complementary, "corrected": corrected,
            "complementary_error": float(errors[0]), "corrected_error": float(errors[1]),
            "complementary_squared_error": float(squares[0]),
            "corrected_squared_error": float(squares[1]), "b_A": b, "RSS": rss,
            "trace_S_transpose_S": trace, "normal_equation_error": normal_error}


def simulate_replicate(item):
    n, d, rep, seed = item
    sizes = sketch_sizes(n, d)
    rows = []
    try:
        data = generate_data(n, d, rep, seed, noise="gaussian")
        data_error = None
    except ERRORS as exc:
        data_error = exc
    for family in FAMILIES:
        try:
            if data_error is not None:
                raise data_error
            prepared = prepare_family(data, sizes, n, d, rep, seed, family)
            preparation_error = None
        except ERRORS as exc:
            preparation_error = exc
        for m in sizes:
            record = _base_record(n, d, m, rep, seed, family)
            try:
                if preparation_error is not None:
                    raise preparation_error
                record.update(fit_family(data, prepared, m), status="ok")
            except ERRORS as exc:
                record.update(failure_type=type(exc).__name__, failure_message=str(exc))
            rows.append(record)
    return rows


def summarize(raw, repetitions):
    """Pointwise paired log-delta 95% intervals, without nonapplicable theory."""
    rows = []
    z975 = norm.ppf(.975)
    for (n, d, m, family), group in raw.groupby(["n", "d", "m", "sketch"], sort=True):
        n, d, m = int(n), int(d), int(m)
        completed = int(group.status.eq("ok").sum())
        failed = len(group) - completed
        row = {"n": n, "d": d, "m": m, "gamma": d / n, "rho": m / n,
               "sketch": family, "repetitions_requested": repetitions,
               "repetitions_attempted": len(group), "repetitions_completed": completed,
               "repetitions_failed": failed,
               "status": "ok" if failed == 0 and len(group) == repetitions else "incomplete",
               "theory_scope": "empirical_non_tight_frame",
               "mse_complementary_mc": np.nan, "mse_corrected_mc": np.nan,
               "mse_ratio_mc": np.nan, "mse_ratio_log_se": np.nan,
               "mse_ratio_ci_low": np.nan, "mse_ratio_ci_high": np.nan}
        if row["status"] == "ok":
            sc = group.complementary_squared_error.to_numpy()
            so = group.corrected_squared_error.to_numpy()
            mc, mo = float(sc.mean()), float(so.mean())
            if min(mc, mo) <= 0.:
                raise ValueError("A log-ratio interval requires positive empirical MSEs")
            ratio = mo / mc
            influence = so / mo - sc / mc
            log_se = float(np.std(influence, ddof=1) / np.sqrt(completed))
            row.update(mse_complementary_mc=mc, mse_corrected_mc=mo,
                       mse_ratio_mc=ratio, mse_ratio_log_se=log_se,
                       mse_ratio_ci_low=ratio * np.exp(-z975 * log_se),
                       mse_ratio_ci_high=ratio * np.exp(z975 * log_se))
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_frame(raw, seed, repetitions, require_complete=True):
    keys = ["n", "d", "m", "sketch"]
    expected = {(n, round(n * gamma), m, family) for n in SAMPLE_SIZES for gamma in GAMMAS
                for m in sketch_sizes(n, round(n * gamma)) for family in FAMILIES}
    if set(raw.groupby(keys).groups) != expected:
        raise ValueError("Archive does not match the exact requested grid")
    if len(raw) != len(expected) * repetitions or raw.fit_id.duplicated().any():
        raise ValueError("Incorrect fit count or duplicate fit identity")
    for (n, d, m, family), group in raw.groupby(keys, sort=True):
        n, d, m = int(n), int(d), int(m)
        if len(group) != repetitions or set(group.rep) != set(range(repetitions)):
            raise ValueError(f"Incomplete repetition IDs at {(n, d, m, family)}")
        base = _base_record(n, d, m, 0, seed, family)
        for key in ("study", "seed", "noise", "sketch", "gamma", "rho", "sigma2",
                    "prediction_signal", "covariance", "nominal_offset", "central_setting", "theory_scope"):
            if not group[key].eq(base[key]).all():
                raise ValueError(f"Invalid archive field {key}")
        for key in ("design_draw_id", "data_draw_id", "sketch_draw_id", "fit_id"):
            expected_values = [_base_record(n, d, m, int(rep), seed, family)[key]
                               for rep in group.rep]
            if list(group[key]) != expected_values:
                raise ValueError(f"Incorrect identity field: {key}")
    if not raw.status.isin(["ok", "numerical_failure"]).all():
        raise ValueError("Unknown fit status")
    if require_complete and raw.status.ne("ok").any():
        raise ValueError("Reporting requires every requested fit; failures are retained without replacement")
    ok = raw.status.eq("ok")
    if not np.isfinite(raw.loc[ok, list(NUMERIC)].to_numpy()).all():
        raise ValueError("Nonfinite successful result")
    if not raw.loc[~ok, list(NUMERIC)].isna().all().all():
        raise ValueError("Failed fit contains reported successful values")
    if (raw.loc[ok, ["risk", "complementary", "corrected", "b_A", "RSS", "trace_S_transpose_S"]] < 0).any().any():
        raise ValueError("Negative risk or criterion")
    if raw.loc[ok, "b_A"].le(0).any() or raw.loc[ok, "normal_equation_error"].gt(1.e-9).any():
        raise ValueError("Invalid fit diagnostic")
    haar = ok & raw.sketch.eq("weighted_haar")
    if not np.allclose(raw.loc[haar, "trace_S_transpose_S"], raw.loc[haar, "n"], rtol=1.e-12, atol=1.e-12):
        raise ValueError("Weighted Haar sketch was not trace-normalized")
    for method in ("complementary", "corrected"):
        error = raw.loc[ok, method].to_numpy() - raw.loc[ok, "risk"].to_numpy()
        if (not np.allclose(error, raw.loc[ok, method + "_error"], rtol=1.e-12, atol=1.e-12)
                or not np.allclose(error ** 2, raw.loc[ok, method + "_squared_error"], rtol=1.e-12, atol=1.e-12)):
            raise ValueError("Paired error does not match its fitted risk")
    return expected


def validate_raw(output, require_paper=True):
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    args = metadata["arguments"]
    reps, seed = int(args["reps"]), int(args["seed"])
    if require_paper and (reps != MSE_REPS or seed != DEFAULT_SEED):
        raise ValueError("Paper run requires 1000 reps and seed 2026092601")
    expected_spec = specification(reps, seed)
    # Accept the original descriptive wording in otherwise identical archives.
    legacy_spec = {
        **expected_spec,
        "theory_scope": "Empirical MSE ordering only; tight-frame exact MSE "
                        "and crossover formulas are not applied",
    }
    if (reps < 2 or metadata.get("study") != STUDY
            or metadata.get("specification") not in (expected_spec, legacy_spec)):
        raise ValueError("Invalid threshold-shapes archive")
    artifacts = metadata["artifact_sha256"]
    if set(artifacts) != {"raw.csv.gz", "summary.csv", "failures.csv"}:
        raise ValueError("Archive requires raw, summary, and failures hashes")
    for name, expected_hash in artifacts.items():
        if digest(output / name) != expected_hash:
            raise ValueError(f"Altered artifact: {name}")
    raw = pd.read_csv(output / "raw.csv.gz", float_precision="round_trip")
    expected = _validate_frame(raw, seed, reps)
    failed = pd.read_csv(output / "failures.csv", float_precision="round_trip")
    if len(failed) or list(failed.columns) != list(raw.columns):
        raise ValueError("Failures archive does not match complete production run")
    summary = pd.read_csv(output / "summary.csv", float_precision="round_trip")
    fresh = summarize(raw, reps)
    if len(summary) != len(expected) or list(summary.columns) != list(fresh.columns):
        raise ValueError("Incorrect summary grid or schema")
    for key in fresh:
        if pd.api.types.is_numeric_dtype(fresh[key]):
            if not np.allclose(summary[key], fresh[key], rtol=1.e-13, atol=1.e-13):
                raise ValueError(f"Summary differs from raw fits: {key}")
        elif not summary[key].eq(fresh[key]).all():
            raise ValueError(f"Summary differs from raw fits: {key}")
    counts = metadata["counts"]
    for key, value in {"settings": len(expected), "independent_datasets": 9 * reps,
                       "attempted_fits": len(raw), "completed_fits": len(raw),
                       "failed_fits": 0, "repetitions_per_setting": reps,
                       "complete_for_reporting": True}.items():
        if counts.get(key) != value:
            raise ValueError(f"Incorrect metadata count: {key}")
    return {"counts": counts, "summary": summary}


def make_figure(summary, output):
    setup_plotting()
    import matplotlib.pyplot as plt
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    titles = {"weighted_haar": "Weighted Haar", "dense_gaussian": "Dense Gaussian"}
    for family in FAMILIES:
        fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.1), sharey=True)
        for row_i, gamma in enumerate(GAMMAS):
            for col_i, n in enumerate(SAMPLE_SIZES):
                ax = axes[row_i, col_i]
                d = round(n * gamma)
                group = summary[(summary.n == n) & (summary.d == d) & (summary.sketch == family)].sort_values("rho")
                if len(group) != len(OFFSETS) or not group.status.eq("ok").all():
                    raise ValueError("Figures require all complete settings")
                rho, ratio = group.rho.to_numpy(), group.mse_ratio_mc.to_numpy()
                low, high = group.mse_ratio_ci_low.to_numpy(), group.mse_ratio_ci_high.to_numpy()
                ax.errorbar(rho, ratio, yerr=[ratio - low, high - ratio],
                            fmt="o-", markersize=4, capsize=2.5, linewidth=1.,
                            color="#0072B2", label="Data simulation: paired 95% interval", zorder=4)
                ax.axhline(1., color="#777777", linestyle="-.", linewidth=.8,
                           label="Equal MSE")
                ax.set_title(fr"$d/n={gamma:.2f}$, $n={n}$", fontsize=10)
                ax.set_xlabel(r"$m/n$")
                ax.set_xlim(float(rho.min()) - .015, float(rho.max()) + .015)
                ax.grid(alpha=.2)
                if col_i == 0:
                    ax.set_ylabel(r"$\mathrm{MSE}(\widehat R_o) / "
                                  r"\mathrm{MSE}(\widehat R_c)$", fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        order = [labels.index(label) for label in ("Data simulation: paired 95% interval", "Equal MSE")]
        axes[0, 0].legend([handles[i] for i in order],
                          ["Data simulation:\npaired 95% interval", "Equal MSE"],
                          loc="upper left", fontsize=7.5, framealpha=.97,
                          borderpad=.4, handlelength=1.4, handletextpad=.4,
                          labelspacing=.4)
        fig.suptitle(titles[family], fontsize=12)
        fig.tight_layout(rect=(0., 0., 1., .96))
        fig.savefig(folder / f"threshold_{family}.pdf")
        fig.savefig(folder / f"threshold_{family}.png", dpi=220)
        plt.close(fig)


def _simulate_block(item):
    n, d, start, stop, seed = item
    return pd.DataFrame([row for rep in range(start, stop)
                         for row in simulate_replicate((n, d, rep, seed))])


def _blocks(reps, seed):
    return [(n, round(n * gamma), first, min(first + CHECKPOINT_REPS, reps), seed)
            for n in SAMPLE_SIZES for gamma in GAMMAS
            for first in range(0, reps, CHECKPOINT_REPS)]


def _block_name(item):
    n, d, start, stop, _ = item
    return f"n{n:04d}-d{d:04d}-reps{start:05d}-{stop:05d}.csv.gz"


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    spec = specification(args.reps, args.seed)
    sources = source_hashes()
    env = environment()
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["specification"] != spec or manifest["source_sha256"] != sources
                or manifest["environment"]["packages"] != env["packages"]):
            raise ValueError("Resume requires identical specification, source, and package versions")
        if (output / "metadata.json").exists():
            validate_raw(output, require_paper=(args.reps == MSE_REPS and args.seed == DEFAULT_SEED))
            print(f"Already complete: {output}", flush=True)
            return
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
        manifest = {"specification": spec, "source_sha256": sources,
                    "environment": env, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "blocks": {}, "elapsed_seconds": 0.}
        atomic_json(manifest_path, manifest)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    blocks = _blocks(args.reps, args.seed)
    legal_names = {_block_name(item) for item in blocks}
    if not set(manifest["blocks"]).issubset(legal_names):
        raise ValueError("Unexpected checkpoint specification")
    for name, block in manifest["blocks"].items():
        if digest(checkpoints / name) != block["sha256"]:
            raise ValueError(f"Altered checkpoint: {name}")
    pending = [item for item in blocks if _block_name(item) not in manifest["blocks"]]
    started = time.perf_counter()
    initial_elapsed = manifest["elapsed_seconds"]
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_simulate_block, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                frame = future.result()
                name = _block_name(item)
                destination = checkpoints / name
                temporary = checkpoints / ("." + name + ".tmp")
                frame.to_csv(temporary, index=False, float_format="%.17g", compression="gzip")
                temporary.replace(destination)
                manifest["blocks"][name] = {"sha256": digest(destination), "rows": len(frame)}
                manifest["elapsed_seconds"] = initial_elapsed + time.perf_counter() - started
                atomic_json(manifest_path, manifest)
                print(f"Saved {len(manifest['blocks'])}/{len(blocks)} blocks; "
                      f"{manifest['elapsed_seconds']:.1f} s", flush=True)
    raw = pd.concat([pd.read_csv(checkpoints / _block_name(item), float_precision="round_trip")
                     for item in blocks], ignore_index=True)
    raw = raw.sort_values(["n", "d", "m", "sketch", "rep"]).reset_index(drop=True)
    _validate_frame(raw, args.seed, args.reps, require_complete=False)
    raw.to_csv(output / "raw.csv.gz", index=False, float_format="%.17g")
    summary = summarize(raw, args.reps)
    summary.to_csv(output / "summary.csv", index=False, float_format="%.17g")
    failures = raw[raw.status.ne("ok")]
    failures.to_csv(output / "failures.csv", index=False, float_format="%.17g")
    counts = {"settings": len(SAMPLE_SIZES) * len(GAMMAS) * len(OFFSETS) * len(FAMILIES),
              "independent_datasets": len(SAMPLE_SIZES) * len(GAMMAS) * args.reps,
              "attempted_fits": len(raw), "completed_fits": len(raw) - len(failures),
              "failed_fits": len(failures), "repetitions_per_setting": args.reps,
              "complete_for_reporting": len(failures) == 0,
              "pairing": spec["pairing"], "ratio_interval": "Pointwise 95% paired log-delta interval",
              "scope": spec["theory_scope"]}
    metadata = {"study": STUDY, "arguments": {"reps": args.reps, "seed": args.seed, "jobs": args.jobs},
                "counts": counts, "specification": spec, "source_sha256": sources,
                "environment": env, "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "artifact_sha256": {name: digest(output / name) for name in
                                    ("raw.csv.gz", "summary.csv", "failures.csv")}}
    atomic_json(output / "metadata.json", metadata)
    if len(failures):
        raise RuntimeError(f"Preserved {len(failures)} failed fits; no draws were replaced")
    validate_raw(output, require_paper=(args.reps == MSE_REPS and args.seed == DEFAULT_SEED))
    make_figure(summary, output)
    print(f"Saved {len(raw)} fits from {counts['independent_datasets']} datasets to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=MSE_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true", help="Revalidate raw fits and render without new draws")
    args = parser.parse_args()
    if args.reps < 2 or args.jobs < 1:
        parser.error("--reps must be at least 2 and --jobs must be positive")
    if args.render:
        result = validate_raw(args.output, require_paper=False)
        make_figure(result["summary"], args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
