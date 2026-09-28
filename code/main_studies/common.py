"""Shared population, random streams, and reporting for the approved main studies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomsketch-main-studies-mpl")

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from utils import (Data, NumericalFailure, Sketch, fit_and_evaluate,
                   fit_compressed, make_covariance, make_sketch, scalar_record,
                   seed_rng)

DEFAULT_SEED = 2026092601
DEFAULT_REPS = 3000
SAMPLE_SIZES = (128, 256, 512)
PREDICTION_SIGNAL = 4.0
NOISE_VARIANCE = 1.0


@lru_cache(maxsize=16)
def population(d):
    """A fixed AR(1) population and the same sparse signal rule in every study."""
    covariance = make_covariance(int(d), "toeplitz")
    chol = np.linalg.cholesky(covariance)
    direction = np.zeros(d)
    direction[:min(10, d)] = 1.0
    beta = np.sqrt(PREDICTION_SIGNAL / (direction @ covariance @ direction)) * direction
    chol.setflags(write=False)
    beta.setflags(write=False)
    return covariance, chol, beta


def generate_data(n, d, rep, seed, noise="gaussian"):
    """Generate the original linear model, paired across methods and studies.

    The feature stream is independent of the error stream. Noise families
    share features, and all sketch sizes share the dataset for this replicate.
    No fitted-risk or residual distribution is used as a data generator.
    """
    covariance, chol, beta = population(d)
    z = seed_rng(seed, "features", n, d, rep).standard_normal((n, d))
    x = z @ chol.T
    rng = seed_rng(seed, "errors", n, d, rep, noise)
    if noise == "gaussian":
        error = rng.standard_normal(n)
    elif noise == "rademacher":
        error = rng.choice([-1.0, 1.0], size=n)
    else:
        raise ValueError(f"Unapproved error distribution: {noise}")
    return Data(x, x @ beta + error, beta, covariance, NOISE_VARIANCE)


def row_sketch(n, m, d, rep, seed):
    """Nested, trace-normalized row sketches independent of all model data."""
    order = seed_rng(seed, "row-permutation", n, d, rep).permutation(n)
    return Sketch(n, m, "row", indices=order[:m],
                  row_scale=np.full(m, np.sqrt(n / m)))


def weighted_sketch(n, m, d, rep, seed):
    return make_sketch(n, m, "weighted",
                       seed_rng(seed, "weighted-haar", n, d, m, rep),
                       normalize=True, weight_condition=2.0)


def run_parallel(fn, items, jobs):
    """Ordered results and one BLAS thread per worker; no randomized redraws."""
    items = list(items)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    count = len(items)
    step = max(1, (count + 19) // 20)
    start = time.perf_counter()
    results = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for i, result in enumerate(executor.map(fn, items), 1):
                results.append(result)
                if i % step == 0 or i == count:
                    print(f"Completed {i}/{count} tasks in "
                          f"{time.perf_counter() - start:.1f} s", flush=True)
    return results


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
    })


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_metadata(out, study, args, counts, elapsed, *, population_info=None,
                  seed_scheme=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for p in Path(__file__).parent.glob("*.py")
                          if not p.name.startswith("._")) + [CODE_DIR / "utils.py"]
    packages = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = {
        "study": study, "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) if hasattr(args, "__dict__") else args,
        "counts": counts, "elapsed_seconds": float(elapsed),
        "population": {"design": "Gaussian", "covariance": "AR(1)",
                       "ar1_correlation": 0.5, "noise_variance": NOISE_VARIANCE,
                       "prediction_signal": PREDICTION_SIGNAL,
                       "signal_direction": "first min(10,d) coordinates equal; others zero"},
        "seed_scheme": ("SHA256 keyed streams for independently frozen summaries and fresh complementary data"
                        if study in ("conditional", "complementary_calibration") else
                        "SHA256 keyed streams; features and noise paired across accuracy and threshold studies; independent sketch streams"),
        "slurm": {key: os.environ.get(key) for key in
                  ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST")},
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "packages": packages, "threadpools": threadpool_info(),
        "source_sha256": {str(path.relative_to(CODE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
    }
    if population_info is not None:
        metadata["population"] = population_info
    if seed_scheme is not None:
        metadata["seed_scheme"] = seed_scheme
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_value) + "\n")
    return metadata
