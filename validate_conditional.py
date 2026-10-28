#!/usr/bin/env python3
"""Independently validate the n=300, m=150 conditional-calibration run.

Usage: python validate_conditional.py --report RUN_DIRECTORY [--paper]
The run directory contains metadata.json, raw.csv.gz, the three summary
CSVs, frozen NPZ files, and atomic per-fit CSV/JSON checkpoints. This helper
imports no study modules and independently recomputes all reported metrics.
It writes validation.json and exits nonzero on any failed check. Without
--paper, the same six-setting grid accepts smaller replication counts from
the run metadata, for smoke tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, t


N, M = 300, 150
DIMENSIONS = (100, 300, 500)
FAMILIES = ("row", "gaussian")
PAPER_SEED, PAPER_FITS, PAPER_DRAWS = 2026092601, 100, 1000
STUDY, KEY_VERSION, BATCH_SIZE = "complementary_calibration", "n300-m150-v1", 25
SETTING_KEYS = ("setting_id", "n", "m", "d", "family", "complement_df", "regime")
METRICS = ("mean_ratio", "variance_ratio", "coverage")
AUDITS = ("orthogonality_error", "summary_error", "refit_error",
          "complement_residual_error", "full_residual_error")
RTOL, ATOL, AUDIT_LIMIT = 5e-11, 5e-12, 1e-8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def finite(values, name):
    array = np.asarray(values, dtype=float)
    require(np.isfinite(array).all(), f"Nonfinite values in {name}")
    return array


def close(actual, expected, name, rtol=RTOL, atol=ATOL):
    actual, expected = finite(actual, name), finite(expected, name + " reference")
    require(actual.shape == expected.shape, f"Wrong shape for {name}: {actual.shape} != {expected.shape}")
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"Numerical mismatch in {name}")


def integer(value, name):
    require(not isinstance(value, (bool, np.bool_)), f"Boolean instead of integer: {name}")
    number = float(value)
    require(np.isfinite(number) and number.is_integer(), f"Noninteger {name}: {value}")
    return int(number)


def booleans(values, name):
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, str) and value.lower() in ("true", "false"):
            result.append(value.lower() == "true")
        else:
            raise ValueError(f"Invalid boolean in {name}: {value!r}")
    return np.asarray(result, dtype=bool)


def catalog():
    return [dict(setting_id=f"d{d:03d}-{family}", n=N, m=M, d=d,
                 family=family, complement_df=N-M,
                 regime="d < m" if d < M else "d > m")
            for d in DIMENSIONS for family in FAMILIES]


def safe_file(root, relative, expected):
    require(str(relative) == expected, f"Unexpected artifact path {relative!r}; expected {expected!r}")
    path = root / expected
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()),
            f"Missing or out-of-report artifact: {expected}")
    return path


def keyed_rng(seed, *keys):
    """Independent implementation of the documented SHA256 stream rule."""
    payload = json.dumps([int(seed), *keys], sort_keys=True, default=str).encode()
    words = np.frombuffer(hashlib.sha256(payload).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def compare_frames(actual, expected, keys, name, exact=False):
    require(set(actual.columns) == set(expected.columns), f"Wrong columns in {name}")
    require(len(actual) == len(expected), f"Wrong row count in {name}")
    require(not actual.duplicated(list(keys)).any(), f"Duplicate row identities in {name}")
    actual = actual.sort_values(list(keys)).reset_index(drop=True)
    expected = expected.sort_values(list(keys)).reset_index(drop=True)
    for column in expected.columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_bool_dtype(right):
            require(np.array_equal(booleans(left, name + "." + column), right.to_numpy()),
                    f"Boolean mismatch in {name}.{column}")
        elif pd.api.types.is_numeric_dtype(right):
            if exact:
                require(np.array_equal(finite(left, name + "." + column), right.to_numpy(dtype=float)),
                        f"Checkpoint values differ in {name}.{column}")
            else:
                close(left.to_numpy(dtype=float), right.to_numpy(dtype=float), name + "." + column)
        else:
            require(left.astype(str).equals(right.astype(str)), f"String mismatch in {name}.{column}")


def check_frozen(path, diagnostic, setting, seed, index, group):
    """Check stored model geometry and replay the first Gaussian data batch."""
    n, m, d, q = setting["n"], setting["m"], setting["d"], setting["complement_df"]
    shapes = {"S": (m, n), "Q": (m, n), "Q_perp": (q, n),
              "X_row": (m, d), "eps_row": (m,), "y_row": (m,),
              "compressed_x": (m, d), "compressed_y": (m,),
              "Sigma": (d, d), "beta": (d,), "beta_hat": (d,),
              "row_residual": (m,), "risk": ()}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(shapes), f"Wrong NPZ fields: {path.name}")
        arrays = {name: finite(archive[name], path.name + ":" + name) for name in shapes}
    for name, shape in shapes.items():
        require(arrays[name].shape == shape, f"Wrong NPZ shape {path.name}:{name}")
    s, f, fc = arrays["S"], arrays["Q"], arrays["Q_perp"]
    sigma, beta, estimate = arrays["Sigma"], arrays["beta"], arrays["beta_hat"]
    coordinate = np.arange(d)
    expected_sigma = .5 ** np.abs(coordinate[:, None] - coordinate[None, :])
    close(sigma, expected_sigma, path.name + ":population covariance")
    direction = np.zeros(d)
    direction[:10] = 1.
    expected_beta = np.sqrt(4. / (direction @ expected_sigma @ direction)) * direction
    close(beta, expected_beta, path.name + ":positive sparse population coefficient")
    close(np.asarray(beta @ sigma @ beta), np.asarray(4.), path.name + ":signal variance")
    geometry_error = max(float(np.linalg.norm(f @ f.T-np.eye(m), ord="fro")/np.sqrt(m)),
                         float(np.linalg.norm(fc @ fc.T-np.eye(q), ord="fro")/np.sqrt(q)),
                         float(np.linalg.norm(f @ fc.T, ord="fro")/np.sqrt(m)))
    require(geometry_error <= AUDIT_LIMIT, f"Invalid orthogonal decomposition: {path.name}")
    annihilation_error = float(np.linalg.norm(s @ fc.T)/max(1., np.linalg.norm(s)))
    require(annihilation_error <= AUDIT_LIMIT, f"Complement is outside ker(S): {path.name}")
    close(s @ f.T @ f, s, path.name + ":retained sketch geometry")
    close(arrays["y_row"], arrays["X_row"] @ beta + arrays["eps_row"], path.name + ":row responses")
    close(arrays["compressed_x"], (s @ f.T) @ arrays["X_row"], path.name + ":literal compressed design")
    close(arrays["compressed_y"], (s @ f.T) @ arrays["y_row"], path.name + ":literal compressed response")
    residual = arrays["y_row"]-arrays["X_row"] @ estimate
    close(arrays["row_residual"], residual, path.name + ":row-space residual")
    row_sse = float(residual @ residual)
    close(np.asarray(diagnostic["row_residual_sse"]), np.asarray(row_sse), path.name + ":row-space SSE")
    delta = estimate-beta
    model_risk = float(1. + delta @ sigma @ delta)
    require(model_risk > 0., f"Nonpositive model risk: {path.name}")
    close(arrays["risk"], np.asarray(model_risk), path.name + ":exact model risk")
    close(np.asarray(diagnostic["risk"]), np.asarray(model_risk), path.name + ":diagnostic risk")
    close(group.risk.to_numpy(dtype=float), np.full(len(group), model_risk), path.name + ":raw fixed risk")
    close(group.row_residual_sse.to_numpy(dtype=float), np.full(len(group), row_sse), path.name + ":raw row SSE")
    close(np.asarray(diagnostic["prediction_signal"]), np.asarray(4.), path.name + ":diagnostic signal")
    b, c = arrays["compressed_x"], arrays["compressed_y"]
    if d < m:
        _, triangular = np.linalg.qr(b, mode="reduced")
        fit_error = float(np.linalg.norm(b.T @ (c-b @ estimate)) /
                          max(1., np.linalg.norm(b)*np.linalg.norm(c)))
    else:
        basis, triangular = np.linalg.qr(b.T, mode="reduced")
        fit_error = max(float(np.linalg.norm(c-b @ estimate)/max(1., np.linalg.norm(c))),
                        float(np.linalg.norm(estimate-basis @ (basis.T @ estimate)) /
                              max(1., np.linalg.norm(estimate))))
    rank_tolerance = np.finfo(float).eps * max(b.shape) * np.linalg.norm(b)
    require(np.min(np.abs(np.diag(triangular))) > rank_tolerance,
            f"Numerically rank-deficient compressed design: {path.name}")
    require(fit_error <= AUDIT_LIMIT, f"Stored coefficient is not the specified fit: {path.name}")
    require(float(diagnostic["compressed_condition"]) >= 1., f"Invalid condition diagnostic: {path.name}")
    key = (STUDY, KEY_VERSION, n, m, d, setting["family"], index)
    sketch_rng = keyed_rng(seed, *key, "sketch")
    if setting["family"] == "row":
        indices = sketch_rng.choice(n, m, replace=False)
        expected_s = np.zeros((m, n))
        expected_s[np.arange(m), indices] = np.sqrt(n/m)
    else:
        expected_s = sketch_rng.standard_normal((m, n))/np.sqrt(m)
    # Row indices and weights replay exactly. Gaussian transforms can differ
    # by a few float64 rounding units across platforms/library versions;
    # require a tight relative bound while keeping artifact hashes exact.
    sketch_rtol = 0. if setting["family"] == "row" else 4*np.finfo(float).eps
    close(s, expected_s, path.name + ":keyed literal sketch", rtol=sketch_rtol, atol=0.)
    sketch_relative_error = float(np.max(np.abs(s-expected_s) /
                                        np.maximum(np.abs(expected_s), np.finfo(float).tiny)))
    chol = np.linalg.cholesky(sigma)
    batch_rng = keyed_rng(seed, *key, "complement-data", 0)
    latent = batch_rng.standard_normal((BATCH_SIZE, q, d))
    noise = batch_rng.standard_normal((BATCH_SIZE, q))
    first_residuals = noise-latent @ (chol.T @ delta)
    count = min(BATCH_SIZE, len(group))
    replay_sse = np.array([float(value @ value) for value in first_residuals[:count]])
    close(group.sort_values("rep").complementary_sse.to_numpy(dtype=float)[:count], replay_sse,
          path.name + ":first keyed Gaussian block batch")
    risk_error = abs(float(arrays["risk"])-model_risk)/max(1., model_risk)
    return {"geometry_error": geometry_error, "annihilation_error": annihilation_error,
            "keyed_sketch_relative_error": sketch_relative_error,
            "fit_optimality_error": fit_error, "model_risk_relative_error": risk_error,
            "replayed_blocks": count}


def validate_report(report, paper=False):
    """Return independent validation evidence; this callable does not write files."""
    report = Path(report).resolve()
    require(report.is_dir(), f"Missing report directory: {report}")
    metadata = json.loads((report / "metadata.json").read_text())
    require(metadata.get("study") == STUDY, "Wrong study identifier")
    arguments, counts = metadata["arguments"], metadata["counts"]
    draws = integer(arguments["reps"], "inner draws")
    fits = integer(arguments["summaries"], "outer frozen fits")
    seed = integer(arguments["seed"], "seed")
    require(draws >= 3 and fits >= 2 and seed >= 0, "Invalid replication or seed specification")
    if paper:
        require((draws, fits, seed) == (PAPER_DRAWS, PAPER_FITS, PAPER_SEED),
                "Paper validation requires 100 fits, 1000 draws and seed 2026092601")
    expected_catalog = catalog()
    observed_catalog = metadata["counts"].get("catalog")
    require(isinstance(observed_catalog, list) and len(observed_catalog) == 6, "Missing six-setting catalog")
    expected_by_id = {item["setting_id"]: item for item in expected_catalog}
    require(len({item["setting_id"] for item in observed_catalog}) == 6, "Duplicate setting catalog")
    require({item["setting_id"] for item in observed_catalog} == set(expected_by_id), "Incorrect setting catalog")
    for item in observed_catalog:
        require(item == expected_by_id[item["setting_id"]], f"Incorrect grid: {item}")
    require(KEY_VERSION in metadata.get("seed_scheme", ""), "Missing versioned random-stream specification")
    population = metadata["population"]
    require(population["design"] == "Gaussian" and population["covariance"] == "AR(1)", "Wrong population model")
    for name, value in (("ar1_correlation", .5), ("noise_variance", 1.), ("prediction_signal", 4.)):
        close(np.asarray(population[name]), np.asarray(value), "metadata population:" + name)
    total_fits, total_draws = 6*fits, 6*fits*draws
    expected_counts = {"settings": 6, "frozen_summaries": total_fits,
                       "complementary_draws": total_draws, "successful_draws": total_draws,
                       "failed_draws": 0, "reconstruction_audit_refits": total_fits}
    for name, expected in expected_counts.items():
        require(integer(counts[name], "metadata count:" + name) == expected, f"Incorrect metadata count: {name}")
    expected_ids = {f"{setting['setting_id']}-summary-{index:03d}": (setting, index)
                    for setting in expected_catalog for index in range(fits)}
    raw = read_csv(report / "raw.csv.gz")
    diagnostics = read_csv(report / "frozen_diagnostics.csv")
    frozen_summaries = read_csv(report / "frozen_summaries.csv")
    settings = read_csv(report / "settings.csv")
    failures = read_csv(report / "failures.csv")
    require(len(raw) == total_draws and len(diagnostics) == total_fits and
            len(frozen_summaries) == total_fits and len(settings) == 6, "Incomplete output counts")
    require(len(failures) == 0, "Failed draws are retained; this report cannot pass completion validation")
    require(raw.status.eq("ok").all() and diagnostics.status.eq("ok").all(), "Run contains failed draws or frozen fits")
    require(raw.error.astype(str).eq("").all() and diagnostics.error.astype(str).eq("").all(), "Successful output has error text")
    require(not raw.duplicated(["summary_id", "rep"]).any(), "Duplicate fit/draw identities")
    for name, frame in (("raw", raw), ("diagnostics", diagnostics), ("frozen summaries", frozen_summaries)):
        require(set(frame.summary_id) == set(expected_ids), f"Wrong fit identity set in {name}")
    require(not diagnostics.summary_id.duplicated().any() and not frozen_summaries.summary_id.duplicated().any(),
            "Duplicate frozen-fit diagnostics or summaries")
    require(booleans(frozen_summaries.complete, "frozen summary completeness").all() and
            booleans(settings.complete, "setting completeness").all(), "Incomplete summaries")
    require(raw.seed.eq(seed).all() and diagnostics.seed.eq(seed).all(), "Wrong random seed in raw records")
    numeric_raw = ("risk", "complementary_sse", "row_residual_sse", "full_residual_sse",
                   "complementary_risk_estimate", "ratio", "ci_lower", "ci_upper")
    for column in numeric_raw:
        values = finite(raw[column], "raw:" + column)
        require((values >= 0.).all(), f"Negative raw quantity: {column}")
    require((raw.risk > 0.).all(), "Nonpositive target risk")
    for name in AUDITS:
        values = finite(diagnostics[name], "audit:" + name)
        require(((values >= 0.) & (values <= AUDIT_LIMIT)).all(), f"Stored audit exceeds 1e-8: {name}")
    quantile_low, quantile_high = chi2.ppf([.025, .975], N-M)
    close(raw.complementary_risk_estimate.to_numpy(dtype=float),
          raw.complementary_sse.to_numpy(dtype=float)/(N-M), "raw SSE-to-estimate algebra")
    close(raw.ratio.to_numpy(dtype=float), raw.complementary_risk_estimate.to_numpy(dtype=float)/raw.risk.to_numpy(dtype=float),
          "raw ratio algebra")
    close(raw.full_residual_sse.to_numpy(dtype=float), raw.complementary_sse.to_numpy(dtype=float)+raw.row_residual_sse.to_numpy(dtype=float),
          "raw full-residual decomposition")
    lower, upper = raw.complementary_sse.to_numpy(dtype=float)/quantile_high, raw.complementary_sse.to_numpy(dtype=float)/quantile_low
    close(raw.ci_lower.to_numpy(dtype=float), lower, "chi-square lower endpoints")
    close(raw.ci_upper.to_numpy(dtype=float), upper, "chi-square upper endpoints")
    require(np.array_equal(booleans(raw.coverage, "raw coverage"), (lower <= raw.risk) & (raw.risk <= upper)),
            "Incorrect chi-square coverage flags")
    expected_csvs = {f"{fit_id}.csv.gz" for fit_id in expected_ids}
    expected_jsons = {f"{fit_id}.json" for fit_id in expected_ids}
    require({p.name for p in (report / "checkpoints").glob("*.csv.gz")} == expected_csvs, "Wrong checkpoint CSV file set")
    require({p.name for p in (report / "checkpoints").glob("*.json")} == expected_jsons, "Wrong checkpoint JSON file set")
    require({p.name for p in (report / "frozen").glob("*.npz")} == {f"{fit_id}.npz" for fit_id in expected_ids},
            "Wrong frozen NPZ file set")
    diagnostic_by_id = diagnostics.set_index("summary_id", drop=False)
    recomputed_fits = []
    artifact_hashes = {}
    largest = {name: 0. for name in ("geometry_error", "annihilation_error", "fit_optimality_error",
                                    "model_risk_relative_error", "keyed_sketch_relative_error")}
    replayed = 0
    for fit_id, group in raw.groupby("summary_id", sort=True):
        setting, index = expected_ids[fit_id]
        require(len(group) == draws, f"Wrong draw count: {fit_id}")
        require(np.array_equal(np.sort(finite(group.rep, fit_id + ":rep")), np.arange(draws)), f"Missing or invalid draw identities: {fit_id}")
        for field, expected in {**setting, "summary_id": fit_id, "summary_index": index, "seed": seed}.items():
            require(group[field].eq(expected).all(), f"Changing or incorrect {field} within {fit_id}")
        require(group.risk.nunique() == 1 and group.row_residual_sse.nunique() == 1, f"Unfrozen risk or retained residual: {fit_id}")
        diagnostic = diagnostic_by_id.loc[fit_id].to_dict()
        for field, expected in {**setting, "summary_index": index, "seed": seed}.items():
            require(diagnostic[field] == expected, f"Incorrect diagnostic {field}: {fit_id}")
        csv_path = safe_file(report, diagnostic["checkpoint_file"], f"checkpoints/{fit_id}.csv.gz")
        npz_path = safe_file(report, diagnostic["frozen_file"], f"frozen/{fit_id}.npz")
        json_path = safe_file(report, f"checkpoints/{fit_id}.json", f"checkpoints/{fit_id}.json")
        csv_hash, npz_hash = digest(csv_path), digest(npz_path)
        require(csv_hash == diagnostic["checkpoint_sha256"], f"Checkpoint CSV hash mismatch: {fit_id}")
        require(npz_hash == diagnostic["frozen_sha256"], f"Frozen NPZ hash mismatch: {fit_id}")
        checkpoint = read_csv(csv_path)
        compare_frames(checkpoint, group, ["summary_id", "rep"], fit_id + ":checkpoint rows", exact=True)
        stored_diagnostic = json.loads(json_path.read_text())
        require(set(stored_diagnostic) == set(diagnostic), f"Checkpoint diagnostic columns differ: {fit_id}")
        for field, expected in diagnostic.items():
            actual = stored_diagnostic[field]
            if isinstance(expected, (int, float, np.number)):
                close(np.asarray(actual), np.asarray(expected), fit_id + ":diagnostic JSON:" + field)
            else:
                require(actual == expected, f"Diagnostic JSON disagreement: {fit_id}:{field}")
        frozen_checks = check_frozen(npz_path, diagnostic, setting, seed, index, group)
        for name in largest:
            largest[name] = max(largest[name], frozen_checks[name])
        replayed += frozen_checks["replayed_blocks"]
        for path, sha in ((csv_path, csv_hash), (npz_path, npz_hash), (json_path, digest(json_path))):
            artifact_hashes[str(path.relative_to(report))] = sha
        ratios = group.ratio.to_numpy(dtype=float)
        row = {**setting, "summary_id": fit_id, "summary_index": index,
               "attempted_draws": draws, "successful_draws": draws, "failed_draws": 0, "complete": True,
               "mean_ratio": float(np.mean(ratios)), "variance_ratio": float(np.var(ratios, ddof=1)/(2/(N-M))),
               "coverage": float(np.mean(booleans(group.coverage, fit_id + ":coverage"))),
               "mean_ratio_mcse": float(np.std(ratios, ddof=1)/np.sqrt(draws)),
               "relative_mse": float(np.mean(np.square(ratios-1))), "risk": float(group.risk.iloc[0])}
        recomputed_fits.append(row)
    recomputed_fits = pd.DataFrame(recomputed_fits)
    compare_frames(frozen_summaries, recomputed_fits, ["summary_id"], "per-frozen-fit metrics")
    recomputed_settings = []
    for setting in expected_catalog:
        group = recomputed_fits.loc[recomputed_fits.setting_id.eq(setting["setting_id"])]
        require(len(group) == fits and set(group.summary_index) == set(range(fits)), "Incorrect outer fit identities")
        row = {**setting, "attempted_summaries": fits, "complete_summaries": fits,
               "attempted_draws": fits*draws, "failed_draws": 0, "complete": True, "cluster_df": fits-1,
               "exact_mean_ratio": 1., "exact_variance_ratio": 1., "exact_coverage": .95,
               "exact_normalized_variance": 2/(N-M)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean, se = float(np.mean(values)), float(np.std(values, ddof=1)/np.sqrt(fits))
            margin = float(t.ppf(.975, fits-1)*se)
            row.update({metric: mean, metric+"_mcse": se, metric+"_low": mean-margin, metric+"_high": mean+margin})
        recomputed_settings.append(row)
    recomputed_settings = pd.DataFrame(recomputed_settings)
    compare_frames(settings, recomputed_settings, ["setting_id"], "outer Student-t summaries")
    for name in ("metadata.json", "raw.csv.gz", "frozen_diagnostics.csv", "frozen_summaries.csv", "settings.csv", "failures.csv"):
        artifact_hashes[name] = digest(report / name)
    return {"status": "passed", "paper": bool(paper), "report": str(report),
            "parameters": {"n": N, "m": M, "d": list(DIMENSIONS), "families": list(FAMILIES),
                           "seed": seed, "fits_per_setting": fits, "draws_per_fit": draws},
            "counts": expected_counts, "outer_interval_df": fits-1, "chi_square_df": N-M,
            "checks": {"grid_and_all_fit_draw_identities": True, "fixed_target_risk": True,
                       "raw_SSE_estimate_ratio_chi_square_coverage_algebra": True,
                       "independently_recomputed_fit_metrics_and_outer_Student_t_intervals": True,
                       "all_stored_audits_finite_at_most_1e_8": True,
                       "all_checkpoint_hashes_rows_and_diagnostic_JSON_agree": True,
                       "all_frozen_NPZ_hashes_shapes_model_risks_and_compressed_data": True,
                       "keyed_sketches_and_first_25_Gaussian_blocks_replayed": True},
            "maximum_stored_audits": {name: float(diagnostics[name].max()) for name in AUDITS},
            "maximum_independent_errors": largest, "independently_replayed_complementary_blocks": replayed,
            "no_new_simulation_fits": True, "summary_functions_imported_or_called": False,
            "validation_environment": {"numpy": np.__version__, "pandas": pd.__version__,
                                       "python": platform.python_version(), "platform": platform.platform()},
            "sketch_replay_policy": {"row": "bitwise exact", "gaussian_relative_tolerance": 4*np.finfo(float).eps,
                                     "gaussian_absolute_tolerance": 0., "artifact_hashes": "exact SHA256"},
            "artifact_sha256": artifact_hashes,
            "validator_source_sha256": digest(Path(__file__))}


def write_validation(report, result):
    temporary = Path(report) / ".validation.tmp.json"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(Path(report) / "validation.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_report(args.report, paper=args.paper)
    except Exception as error:
        if args.report.is_dir():
            write_validation(args.report, {"status": "failed", "paper": bool(args.paper),
                             "error": f"{type(error).__name__}: {error}",
                             "validator_source_sha256": digest(Path(__file__))})
        raise
    write_validation(args.report, result)
    print(json.dumps({"status": result["status"], "paper": result["paper"],
                      "counts": result["counts"], "outer_interval_df": result["outer_interval_df"],
                      "chi_square_df": result["chi_square_df"],
                      "replayed_blocks": result["independently_replayed_complementary_blocks"]}))


if __name__ == "__main__":
    main()
