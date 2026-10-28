#!/usr/bin/env python3
"""Run the current manuscript studies or rebuild their figures and tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "code" / "main_studies"
STUDIES = {
    "conditional": {
        "seed": 2026092601, "reps": 1000, "input": "settings.csv", "rows": 6,
        "figure": "complementary_calibration", "extra": ["--summaries", "100"],
        "counts": {"settings": 6, "frozen_summaries": 600,
                   "complementary_draws": 600000, "successful_draws": 600000,
                   "failed_draws": 0},
    },
    "risk_curves": {
        "seed": 2026100901, "reps": 1000, "extra": [],
    },
    "threshold": {
        "seed": 2026092601, "reps": 1000, "input": "summary.csv", "rows": 45,
        "figure": "threshold", "extra": [],
        "counts": {"settings": 45, "independent_datasets": 9000,
                   "attempted_fits": 45000, "completed_fits": 45000,
                   "failed_fits": 0},
    },
    "threshold_shapes": {
        "seed": 2026092601, "reps": 1000, "input": "summary.csv", "rows": 90,
        "figure": "threshold_shapes", "extra": [],
        "counts": {"settings": 90, "independent_datasets": 9000,
                   "attempted_fits": 90000, "completed_fits": 90000,
                   "failed_fits": 0},
    },
}
CURRENT_STUDIES = ("conditional", "risk_curves", "threshold", "threshold_shapes")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def source_hashes():
    paths = [ROOT / "reproduce.py", ROOT / "validate_conditional.py", ROOT / "code" / "utils.py"]
    paths.extend(sorted(SOURCE.glob("*.py")))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def selected(name):
    return list(CURRENT_STUDIES) if name == "all" else [name]


def require_fresh(paths):
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Output is not empty: {path}. Choose a new --output-root.")


def check_completed(name, folder):
    """Reject incomplete runs before rendering the published summaries."""
    import pandas as pd

    spec = STUDIES[name]
    if name == "risk_curves":
        sys.path.insert(0, str(SOURCE))
        checked = importlib.import_module("risk_curves").validate_raw(folder, require_paper=True)
        if checked["specification"]["seed"] != spec["seed"]:
            raise ValueError("risk_curves: results do not use the manuscript seed")
        return checked
    metadata = json.loads((folder / "metadata.json").read_text())
    arguments, counts = metadata["arguments"], metadata["counts"]
    expected_reps = spec["reps"]
    if arguments.get("reps") != expected_reps or arguments.get("seed") != spec["seed"]:
        raise ValueError(f"{name}: results do not use the published seed and repetitions")
    if name == "conditional" and arguments.get("summaries") != 100:
        raise ValueError("conditional: the paper requires 100 frozen fits per setting")
    for key, expected in spec["counts"].items():
        if counts.get(key) != expected:
            raise ValueError(f"{name}: {key}={counts.get(key)!r}, expected {expected}")
    if counts.get("complete_for_reporting", True) is not True:
        raise ValueError(f"{name}: simulation marked incomplete")
    frame = pd.read_csv(folder / spec["input"])
    if len(frame) != spec["rows"]:
        raise ValueError(f"{name}: incomplete summary table")
    if "status" in frame and not frame["status"].eq("ok").all():
        raise ValueError(f"{name}: summary includes failed settings")
    if "complete" in frame and not frame["complete"].eq(True).all():
        raise ValueError(f"{name}: summary includes incomplete settings")
    if "count" in frame and not frame["count"].eq(expected_reps).all():
        raise ValueError(f"{name}: summary has an incomplete repetition count")
    if name == "conditional":
        expected_grid = {(300, 150, d, family) for d in (100, 300, 500)
                         for family in ("row", "gaussian")}
        observed_grid = set(frame[["n", "m", "d", "family"]].itertuples(index=False, name=None))
        if observed_grid != expected_grid or not frame.complete_summaries.eq(100).all():
            raise ValueError("conditional: incorrect grid or frozen-fit count")
        if not frame.attempted_draws.eq(100000).all():
            raise ValueError("conditional: incorrect complementary-data draw count")
        sys.path.insert(0, str(ROOT))
        importlib.import_module("validate_conditional").validate_report(folder, paper=True)
    if name in ("threshold", "threshold_shapes"):
        sys.path.insert(0, str(SOURCE))
        checked = importlib.import_module(name).validate_raw(folder, require_paper=True)
        return checked["summary"]
    return frame


def child_environment(cache):
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(cache))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def run_studies(args, env):
    names = selected(args.study)
    output = args.output_root.resolve()
    require_fresh([output / name for name in names])
    for name in names:
        spec = STUDIES[name]
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        repetitions = spec["reps"]
        command = [sys.executable, str(SOURCE / f"{name}.py")]
        if name == "risk_curves":
            command.append("run")
        command += ["--output", str(destination), "--reps", str(repetitions),
                   "--seed", str(spec["seed"]), "--jobs", str(args.jobs), *spec["extra"]]
        record = {"started_utc": datetime.now(timezone.utc).isoformat(),
                  "study": name, "command": command, "status": "running",
                  "source_sha256": source_hashes()}
        manifest = destination / "reproduction_run.json"
        # Resumable studies require an initially empty directory.
        if name not in ("risk_curves", "threshold_shapes"):
            save_json(manifest, record)
        print(f"Running {name}: {repetitions:,} repetitions; {args.jobs} workers", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            check_completed(name, destination)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            record.update(status="failed", error=str(error))
            raise
        else:
            record["status"] = "completed"
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(manifest, record)
    print(f"Finished. Figures, tables, raw draws, and metadata: {output}")


def render_studies(args, env):
    names = selected(args.study)
    inputs, output = args.input_root.resolve(), args.output_root.resolve()
    if inputs == output or inputs in output.parents or output in inputs.parents:
        raise ValueError("Input and output roots must be separate, non-nested directories")
    require_fresh([output / name for name in names])
    # Check all selected studies before creating any presentation output.
    frames = {name: check_completed(name, inputs / name) for name in names}
    sys.path.insert(0, str(SOURCE))
    for name in names:
        spec = STUDIES[name]
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        if name == "risk_curves":
            source = inputs / name
            shutil.copytree(source / "cells", destination / "cells")
            shutil.copy2(source / "manifest.json", destination / "manifest.json")
            subprocess.run([sys.executable, str(SOURCE / "risk_curves.py"),
                            "render", "--output", str(destination)],
                           cwd=ROOT, env=env, check=True)
            save_json(destination / "render_manifest.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "operation": "Validate complete raw cells and recompute summaries, figures and tables; no new draws",
                "input_directory": str(source),
                "input_sha256": {str(Path(path).relative_to(source)): digest(path)
                                 for path in frames[name]["cell_csvs"]},
                "source_sha256": source_hashes(),
            })
            print(f"Rendered {name}: {destination}", flush=True)
            continue
        files = [spec["input"], "metadata.json"]
        input_hashes = {}
        for filename in files:
            source = inputs / name / filename
            shutil.copy2(source, destination / filename)
            input_hashes[filename] = digest(source)
        module = importlib.import_module(name)
        module.make_figure(frames[name], destination)
        save_json(destination / "render_manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "Render saved summaries; no new simulation draws",
            "input_directory": str(inputs / name), "input_sha256": input_hashes,
            "source_sha256": source_hashes(), "python": platform.python_version(),
            "packages": {package: importlib.metadata.version(package)
                         for package in ("numpy", "scipy", "pandas", "matplotlib")},
        })
        print(f"Rendered {name}: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Generate original data for the four current manuscript studies")
    run.add_argument("--study", choices=["all", *STUDIES], default="all")
    run.add_argument("--output-root", type=Path, default=ROOT / "runs" / "paper")
    run.add_argument("--jobs", type=int, default=1)
    render = subparsers.add_parser("render", help="Rebuild figures/tables; risk_curves summaries are recomputed from raw fits")
    render.add_argument("--study", choices=["all", *STUDIES], default="all")
    render.add_argument("--input-root", type=Path, default=ROOT / "runs" / "paper",
                        help="Completed simulation root (default: runs/paper)")
    render.add_argument("--output-root", type=Path, default=ROOT / "runs" / "rendered")
    args = parser.parse_args()
    if args.command == "run" and args.jobs < 1:
        parser.error("--jobs must be positive")
    with tempfile.TemporaryDirectory(prefix="randomsketch-mpl-") as cache:
        env = child_environment(cache)
        # Imports in rendering and completion checks also use this writable cache.
        os.environ.update({key: value for key, value in env.items()
                           if key.startswith("MPL") or key.endswith("NUM_THREADS")
                           or key == "VECLIB_MAXIMUM_THREADS"})
        try:
            if args.command == "run":
                run_studies(args, env)
            else:
                render_studies(args, env)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            parser.exit(1, f"Reproduction stopped: {error}\n")


if __name__ == "__main__":
    main()
