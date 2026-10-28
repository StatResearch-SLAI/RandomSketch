"""Numerical and reproducibility tools for observation-sketch simulations.

No feature whitening, ridge stabilization, or pseudoinverse truncation is
silently applied. Row sketches are operators so structured application costs
are represented faithfully. Arrays use float64 throughout.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular, svdvals
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

_mpl_cache = Path(tempfile.gettempdir())/'randomsketch-matplotlib'
_mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_mpl_cache))


class NumericalFailure(RuntimeError):
    """A sampled problem could not be evaluated without changing the estimator."""


@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    Sigma: np.ndarray
    sigma2: float
    extra_risk: float = 0.0


def seed_rng(seed, *keys):
    """A stream keyed by experiment/cell/replicate, independent of scheduling."""
    payload = json.dumps([int(seed), *keys], sort_keys=True, default=str).encode()
    words = np.frombuffer(hashlib.sha256(payload).digest(), dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


@lru_cache(maxsize=32)
def make_covariance(d, kind='identity'):
    """Bounded-spectrum covariance families; returned matrices are read-only.

    Equicorrelation uses off-diagonal 0.5/d, avoiding the diverging largest
    eigenvalue of fixed-positive equicorrelation. Spikes have fixed magnitude.
    """
    if d < 1:
        raise ValueError('d must be positive')
    if kind == 'identity':
        cov = np.eye(d)
    elif kind == 'toeplitz':
        ix = np.arange(d)
        cov = 0.5 ** np.abs(ix[:, None] - ix[None, :])
    elif kind == 'spiked':
        values = np.ones(d)
        values[:min(2, d)] = np.array([5.0, 10.0])[:min(2, d)]
        cov = np.diag(values)
    elif kind == 'equicorrelated':
        rho = 0.5 / d
        cov = (1-rho)*np.eye(d) + rho*np.ones((d, d))
    else:
        raise ValueError('Unknown covariance: ' + kind)
    cov.setflags(write=False)
    return cov


@lru_cache(maxsize=32)
def _covariance_factor(d, kind):
    factor = np.linalg.cholesky(make_covariance(d, kind))
    factor.setflags(write=False)
    return factor


def generate_data(n, d, rng, covariance='identity', design='gaussian',
                  noise='gaussian', signal=2.0, nonlinear=0.0,
                  heteroskedastic=False):
    """Generate a fixed-coefficient model and its exact fresh-sample risk target.

    signal is ||beta||_2, not beta' Sigma beta. Nonlinear sensitivity uses
    a centered, unit-variance quadratic of a Gaussian feature; it is orthogonal
    to every linear predictor. Heteroskedastic noise has marginal variance one.
    """
    if min(n, d) < 1 or signal < 0:
        raise ValueError('Positive dimensions and nonnegative signal required')
    cov = make_covariance(d, covariance)
    if design == 'gaussian':
        z = rng.standard_normal((n, d))
    elif design == 'rademacher':
        z = rng.choice([-1.0, 1.0], size=(n, d))
    elif design == 't5':
        z = rng.standard_t(5, size=(n, d)) / np.sqrt(5/3)
    else:
        raise ValueError('Unknown design: ' + design)
    X = z if covariance == 'identity' else z @ _covariance_factor(d, covariance).T
    beta = np.zeros(d)
    s = max(1, d // 5)
    beta[:s] = signal / np.sqrt(s)
    if noise == 'gaussian':
        eps = rng.standard_normal(n)
    elif noise == 'rademacher':
        eps = rng.choice([-1.0, 1.0], size=n)
    elif noise == 'uniform':
        eps = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)
    elif noise == 'laplace':
        eps = rng.laplace(scale=1/np.sqrt(2), size=n)
    elif noise == 't5':
        eps = rng.standard_t(5, size=n) / np.sqrt(5/3)
    else:
        raise ValueError('Unknown noise: ' + noise)
    if heteroskedastic:
        eps *= np.sqrt((1 + X[:, 0]**2/cov[0, 0])/2)
    y = X @ beta + eps
    if nonlinear:
        if design != 'gaussian':
            raise ValueError('The exact nonlinear target requires Gaussian design')
        y += nonlinear * (X[:, 0]**2/cov[0, 0] - 1) / np.sqrt(2)
    return Data(X, y, beta, cov, 1.0, float(nonlinear**2))


def _fwht(v):
    """Normalized Walsh--Hadamard transform along the first axis."""
    x = np.array(v, dtype=float, copy=True)
    n = x.shape[0]
    if n < 1 or n & (n-1):
        raise ValueError('SRHT requires n to be a power of two')
    shape = x.shape
    x = x.reshape(n, -1)
    h = 1
    while h < n:
        blocks = x.reshape(-1, 2*h, x.shape[1])
        a, b = blocks[:, :h].copy(), blocks[:, h:].copy()
        blocks[:, :h], blocks[:, h:] = a+b, a-b
        h *= 2
    return x.reshape(shape) / np.sqrt(n)


class Sketch:
    """S and its adjoint without an n-by-n A=S.T@S allocation."""
    def __init__(self, n, m, family, matrix=None, indices=None, signs=None,
                 row_scale=None, row_gram_diag=None):
        self.n, self.m, self.family = int(n), int(m), family
        self.matrix, self.indices, self.signs = matrix, indices, signs
        self.row_scale = row_scale
        self.row_gram_diag = (None if row_gram_diag is None else
                              np.asarray(row_gram_diag, dtype=float))
        self.tight_scale = None
        if matrix is None and row_scale is not None and np.all(row_scale == row_scale[0]):
            self.tight_scale = float(row_scale[0]**2)

    @classmethod
    def from_matrix(cls, S, family='custom'):
        S = np.asarray(S, dtype=float)
        if S.ndim != 2 or not (1 <= S.shape[0] <= S.shape[1]):
            raise ValueError('S must be an m by n matrix, 1 <= m <= n')
        return cls(S.shape[1], S.shape[0], family, matrix=S)

    def apply(self, v):
        v = np.asarray(v)
        if v.shape[0] != self.n:
            raise ValueError('Sketch application has incompatible dimensions')
        if self.matrix is not None:
            return self.matrix @ v
        scale = self.row_scale if v.ndim == 1 else self.row_scale[:, None]
        if self.family == 'row':
            return scale*v[self.indices]
        sign = self.signs if v.ndim == 1 else self.signs[:, None]
        return scale*_fwht(sign*v)[self.indices]

    def adjoint(self, v):
        v = np.asarray(v)
        if v.shape[0] != self.m:
            raise ValueError('Sketch adjoint has incompatible dimensions')
        if self.matrix is not None:
            return self.matrix.T @ v
        x = np.zeros((self.n,) + v.shape[1:])
        scale = self.row_scale if v.ndim == 1 else self.row_scale[:, None]
        x[self.indices] = scale*v
        if self.family == 'row':
            return x
        sign = self.signs if v.ndim == 1 else self.signs[:, None]
        return sign*_fwht(x)

    def row_gram(self):
        if self.row_gram_diag is not None:
            return np.diag(self.row_gram_diag)
        if self.matrix is not None:
            return self.matrix @ self.matrix.T
        return np.diag(self.row_scale**2)

    def to_dense(self):
        if self.matrix is not None:
            return self.matrix.copy()
        return self.adjoint(np.eye(self.m)).T


def make_sketch(n, m, family, rng, normalize=False, weight_condition=2.0):
    if not (1 <= m <= n):
        raise ValueError('Require 1 <= m <= n')
    if family in ('row', 'srht'):
        if family == 'srht' and n & (n-1):
            raise ValueError('SRHT requires a power-of-two n')
        ix = rng.choice(n, m, replace=False)
        signs = rng.choice([-1.0, 1.0], n) if family == 'srht' else None
        return Sketch(n, m, family, indices=ix, signs=signs,
                      row_scale=np.full(m, np.sqrt(n/m)))
    if family == 'gaussian':
        S = rng.standard_normal((m, n)) / np.sqrt(m)
    elif family in ('haar', 'weighted', 'ill_conditioned'):
        Q, R = qr(rng.standard_normal((n, m)), mode='economic')
        Q *= np.where(np.diag(R) >= 0, 1.0, -1.0)
        weights = np.ones(m)
        if family == 'weighted':
            if weight_condition < 1:
                raise ValueError('weight_condition must be >= 1')
            weights[1::2] = weight_condition
        elif family == 'ill_conditioned':
            weights = np.geomspace(1.0, 1.0/n, m)
        S = np.sqrt(n/np.dot(weights, weights))*weights[:, None]*Q.T
    else:
        raise ValueError('Unknown sketch family: ' + family)
    if normalize:
        S *= np.sqrt(n/np.sum(S*S))
    sketch = Sketch.from_matrix(S, family)
    if family == 'haar':
        sketch.tight_scale = n/m
    elif family == 'weighted':
        sketch.row_gram_diag = (n/np.dot(weights, weights))*weights**2
    return sketch


def fit_compressed(B, c, diagnostics=True):
    """Full-rank QR solves, including the square interpolation boundary."""
    m, d = B.shape
    if diagnostics:
        sv = svdvals(B, check_finite=True)
        tol = np.finfo(float).eps*max(m, d)*sv[0]
        rank = int(np.count_nonzero(sv > tol))
        condition = float(sv[0]/sv[-1]) if sv[-1] else np.inf
        if rank != min(m, d):
            raise NumericalFailure(f'Compressed design rank {rank} < {min(m,d)}')
    else:
        rank, condition = min(m, d), np.nan
    if d < m:
        q, r = qr(B, mode='economic', check_finite=False)
        beta_hat = solve_triangular(r, q.T @ c, check_finite=False)
        regime = 'noninterpolating'
    else:
        q, r = qr(B.T, mode='economic', check_finite=False)
        beta_hat = q @ solve_triangular(r.T, c, lower=True, check_finite=False)
        regime = 'boundary' if d == m else 'interpolating'
    if not np.all(np.isfinite(beta_hat)):
        raise NumericalFailure('Nonfinite coefficient estimate')
    return dict(beta_hat=beta_hat, regime=regime, rank=rank,
                condition=condition, q=q, r=r)


def realized_risk(data, beta_hat):
    error = beta_hat-data.beta
    return float(data.sigma2 + data.extra_risk + error @ data.Sigma @ error)


def fit_and_evaluate(data, sketch, compute_r1=True, diagnostics=True):
    """Fit once and evaluate the manuscript criteria; truth is timed separately.

    calibration_seconds includes the sketch-weighted residual and b_A needed
    by R_obs (stored in the legacy ``r2`` field). r1_seconds is the additional
    population-covariance trace computation for the unobservable risk estimate
    (stored in the legacy ``R_sens``/``r1`` fields).
    Diagnostics (when requested) include SVD time in fit_seconds.
    """
    n, d = data.X.shape
    m = sketch.m
    start = time.perf_counter()
    B, c = sketch.apply(data.X), sketch.apply(data.y)
    apply_seconds = time.perf_counter()-start
    start = time.perf_counter()
    fit = fit_compressed(B, c, diagnostics)
    fit_seconds = time.perf_counter()-start
    beta_hat = fit['beta_hat']
    start = time.perf_counter()
    u = data.y-data.X @ beta_hat
    sse = float(u @ u)
    residual_seconds = time.perf_counter()-start
    result = {k: fit[k] for k in ('beta_hat', 'regime', 'rank', 'condition')}
    result.update(raw=sse/n, r1=np.nan, r2=np.nan, validation=np.nan,
                  b=np.nan, t=np.nan, df=min(m,d),
                  apply_seconds=apply_seconds, fit_seconds=fit_seconds,
                  residual_seconds=residual_seconds, calibration_seconds=0.0,
                  validation_seconds=0.0, r1_seconds=0.0)
    result['gcv'] = sse/n/(1-min(m,d)/n)**2 if min(m,d) < n else np.nan
    sketched_residual = None
    sketched_residual_seconds = 0.0
    if m < n:
        start = time.perf_counter()
        apply_start = time.perf_counter()
        sketched_residual = sketch.apply(u)
        sketched_residual_seconds = time.perf_counter()-apply_start
        if sketch.tight_scale is not None:
            projected_sse = float(sketched_residual @ sketched_residual /
                                  sketch.tight_scale)
        elif sketch.row_gram_diag is not None:
            projected_sse = float(np.sum(sketched_residual**2 /
                                         sketch.row_gram_diag))
        else:
            row_gram = sketch.row_gram()
            projected_sse = float(sketched_residual @
                                  np.linalg.solve(row_gram, sketched_residual))
        complementary_sse = sse-projected_sse
        tolerance = 100*np.finfo(float).eps*max(1.0, sse, projected_sse)
        if complementary_sse < -tolerance:
            raise NumericalFailure(
                f'Negative complementary residual sum of squares: {complementary_sse}')
        result['validation'] = max(0.0, complementary_sse)/(n-m)
        result['validation_seconds'] = time.perf_counter()-start
    if d < m:
        start = time.perf_counter()
        reused_sketched_residual = sketched_residual is not None
        if sketched_residual is None:
            apply_start = time.perf_counter()
            sketched_residual = sketch.apply(u)
            sketched_residual_seconds = time.perf_counter()-apply_start
        weighted_residual = sketch.adjoint(sketched_residual)
        if sketch.tight_scale is not None:
            b = sketch.tight_scale*(m-d)
        elif sketch.row_gram_diag is not None:
            diagonal = sketch.row_gram_diag
            b = float(np.sum(diagonal)-np.sum(fit['q']**2*diagonal[:, None]))
        else:
            D = sketch.row_gram()
            b = float(np.trace(D)-np.sum(fit['q']*(D @ fit['q'])))
        if not np.isfinite(b) or b <= 0:
            raise NumericalFailure(f'Nonpositive or nonfinite residual sketch mass: {b}')
        corrected = u+(d/b)*weighted_residual
        result.update(b=b, r2=float(corrected @ corrected/n))
        result['calibration_seconds'] = (time.perf_counter()-start+
                                         (sketched_residual_seconds
                                          if reused_sketched_residual else 0.0))
        if compute_r1:
            start = time.perf_counter()
            inverse_r = solve_triangular(fit['r'], np.eye(d), check_finite=False)
            t = float(np.sum(inverse_r*(data.Sigma @ inverse_r)))
            corrected = u+t*weighted_residual
            result.update(t=t, r1=float(corrected @ corrected/n))
            result['r1_seconds'] = time.perf_counter()-start
    start = time.perf_counter()
    result['risk'] = realized_risk(data, beta_hat)
    result['truth_seconds'] = time.perf_counter()-start
    # Numerical diagnostics are not included in assessment costs.
    result['interpolation_error'] = float(np.linalg.norm(B @ beta_hat-c) /
                                           max(1.0, np.linalg.norm(c)))
    return result


def scalar_record(result):
    return {k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in result.items() if np.isscalar(v) or v is None}


def theory_risk(n, d, m, family, signal=2.0, sigma2=1.0):
    """Gaussian benchmarks; interpolating formulas additionally require Sigma=I.

    Caller must enforce distributional scope. Values use realized d/n,m/n.
    Missing finite inverse moments and m=n validation are returned as NaN.
    """
    out = dict(risk_exact=np.nan, risk_limit=np.nan,
               gcv_limit=np.nan, gcv_gap_limit=np.nan)
    gamma, rho = d/n, m/n
    if d < m:
        if family in ('row', 'haar', 'srht', 'tight_frame'):
            if m > d+1:
                out['risk_exact'] = sigma2*(1+d/(m-d-1))
            out['risk_limit'] = sigma2*rho/(rho-gamma)
            out['gcv_gap_limit'] = sigma2*gamma**2*(1-rho)/((rho-gamma)*(1-gamma)**2)
        elif family == 'gaussian':
            if m > d+1:
                out['risk_exact'] = sigma2*(1+d/(n-d-1)*(1+(n-d)/(m-d-1)))
            out['risk_limit'] = sigma2*(1/(1-gamma)+gamma/(rho-gamma))
            out['gcv_gap_limit'] = sigma2*gamma**2/((1-gamma)*(rho-gamma))
        out['gcv_limit'] = out['risk_limit']+out['gcv_gap_limit']
    elif d > m:
        if d > m+1:
            out['risk_exact'] = sigma2+(1-m/d)*signal**2+sigma2*m/(d-m-1)
        out['risk_limit'] = signal**2*(1-rho/gamma)+sigma2*gamma/(gamma-rho)
        if m < n:
            out['gcv_limit'] = out['risk_limit']/(1-rho)
            out['gcv_gap_limit'] = out['gcv_limit']-out['risk_limit']
    return out


def add_common_arguments(parser):
    parser.add_argument('--profile', choices=['smoke', 'quick', 'paper'], default='quick')
    parser.add_argument('--reps', type=int, default=None,
                        help='Override outer Monte Carlo replicates per scenario')
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--no-plots', action='store_true')
    return parser


def prepare_output(args, experiment):
    if args.reps is not None and args.reps < 1:
        raise ValueError('--reps must be positive')
    if args.jobs == 0:
        raise ValueError('--jobs cannot be zero')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output or (Path(__file__).resolve().parent/'results'/experiment /
                          f'{args.profile}-seed{args.seed}-{stamp}')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists() or (out/'raw.csv.gz').exists():
        raise FileExistsError(f'Refusing to overwrite completed output: {out}')
    return out


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_results(rows, out, metadata, summary=None):
    out = Path(out)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    temp = out/'.raw.csv.gz.tmp'
    frame.to_csv(temp, index=False, compression='gzip')
    temp.replace(out/'raw.csv.gz')
    if summary is not None:
        pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
    versions = {}
    for name in ['numpy','scipy','pandas','matplotlib','joblib','threadpoolctl']:
        versions[name] = importlib.metadata.version(name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).resolve().parent.glob('*.py')}
    manuscript = Path(__file__).resolve().parents[1]/'main.tex'
    try:
        manuscript_hash = hashlib.sha256(manuscript.read_bytes()).hexdigest()
    except OSError:
        manuscript_hash = None
    meta = dict(metadata)
    if 'status' in frame:
        meta['status_counts'] = frame.status.fillna('unspecified').value_counts().to_dict()
    meta.update(created_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, platform=platform.platform(), packages=versions,
                source_sha256=hashes, manuscript_sha256=manuscript_hash,
                rows=len(frame), blas_threads_per_worker=1)
    (out/'metadata.json').write_text(json.dumps(_json_safe(meta), indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(frame)} rows to {out}', flush=True)
    return frame


def _task_with_threads(fn, item):
    with threadpool_limits(limits=1):
        return fn(item)


def run_tasks(fn, items, jobs=1):
    """Parallelize independent replicates; BLAS has one thread per worker."""
    items = list(items)
    if jobs == 1:
        out = []
        with threadpool_limits(limits=1):
            for j, item in enumerate(items):
                out.append(fn(item))
                if (j+1) % max(1, len(items)//10) == 0:
                    print(f'Completed {j+1}/{len(items)} tasks', flush=True)
        return out
    return Parallel(n_jobs=jobs, backend='loky', verbose=5)(
        delayed(_task_with_threads)(fn, item) for item in items)


def configure_plotting():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':120,
                         'savefig.dpi':240, 'pdf.fonttype':42,
                         'axes.grid':True, 'grid.alpha':0.2})


def save_figure(fig, out, name):
    folder = Path(out)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder/f'{name}.png', bbox_inches='tight', dpi=240)


def summarize_errors(frame, group_cols, criteria):
    """Bias/MAE/RMSE and MCSE of replicate-level errors; failures are counted."""
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, group in frame.groupby(grouper, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        for criterion in criteria:
            err = ((group[criterion]-group['risk']).to_numpy(dtype=float)
                   if criterion in group else np.full(len(group),np.nan))
            err = err[np.isfinite(err)]
            N = len(err)
            rows.append(dict(base, criterion=criterion, count=N, attempted=len(group),
                             unavailable=len(group)-N, bias=float(np.mean(err)) if N else np.nan,
                             bias_mcse=float(np.std(err,ddof=1)/np.sqrt(N)) if N>1 else np.nan,
                             mae=float(np.mean(np.abs(err))) if N else np.nan,
                             mae_mcse=float(np.std(np.abs(err),ddof=1)/np.sqrt(N)) if N>1 else np.nan,
                             rmse=float(np.sqrt(np.mean(err**2))) if N else np.nan))
    return pd.DataFrame(rows)
