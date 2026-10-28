#!/usr/bin/env python3
"""Original-data Monte Carlo mean-risk curves; complete cells are resumable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache, partial
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomsketch-risk-curves-mpl")
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular
from scipy.signal import lfilter
from threadpoolctl import threadpool_info, threadpool_limits

STUDY = "expanded-original-data-risk-curves-v1"
SCHEMA = 1
SEED = 2026100901
REPS = 1000
NS = tuple(range(100, 1001, 50))
RATIOS = ((.3, .5), (.5, .7), (.7, .9), (.8, 1.))
SKETCHES = ("row", "weighted_haar", "dense_gaussian")
NOISES = ("gaussian", "uniform_minus1_plus1")
VALUES = ("risk", "R_obs", "R_sens", "R_GCV", "R_B", "R_perp")
KEYS = ("n", "d", "m", "gamma", "rho", "sketch", "noise")
CHECKPOINT = 100


class NumericalFailure(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_rng(seed, *keys):
    payload = json.dumps([int(seed), STUDY, *keys], sort_keys=True).encode()
    words = np.frombuffer(hashlib.sha256(payload).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def cells():
    return [dict(cell_index=i, n=n, d=round(n*g), m=round(n*r), gamma=g, rho=r)
            for i, (n, g, r) in enumerate((n, g, r) for n in NS for g, r in RATIOS)]


@lru_cache(maxsize=32)
def population(d):
    ix = np.arange(d)
    cov = .5 ** np.abs(ix[:, None] - ix[None, :])
    L = np.linalg.cholesky(cov)
    direction = np.zeros(d)
    direction[:min(10, d)] = 1.
    beta = np.sqrt(4. / (direction @ cov @ direction)) * direction
    for value in (cov, L, beta):
        value.setflags(write=False)
    return cov, L, beta


def generate_data(n, d, rep, seed):
    """Actual AR(1) feature matrix and independent original error vectors."""
    cov, L, beta = population(d)
    z = seed_rng(seed, "features", n, d, rep).standard_normal((n, d))
    z[:, 0] /= np.sqrt(.75)
    X = lfilter([np.sqrt(.75)], [1., -.5], z, axis=1)
    E = np.column_stack((
        seed_rng(seed, "errors", n, d, rep, NOISES[0]).standard_normal(n),
        seed_rng(seed, "errors", n, d, rep, NOISES[1]).uniform(-1., 1., n)))
    Y = (X @ beta)[:, None] + E
    return X, Y, cov, L, beta


class Sketch:
    def __init__(self, n, m, family, indices=None, matrix=None, diagonal=None):
        self.n, self.m, self.family = n, m, family
        self.indices, self.matrix, self.diagonal = indices, matrix, diagonal

    def apply(self, v):
        return np.sqrt(self.n/self.m)*v[self.indices] if self.family == "row" else self.matrix @ v

    def adjoint(self, v):
        if self.family != "row":
            return self.matrix.T @ v
        result = np.zeros((self.n,) + v.shape[1:])
        result[self.indices] = np.sqrt(self.n/self.m)*v
        return result


def make_sketch(n, d, m, rep, seed, family):
    rng = seed_rng(seed, "sketch", family, n, d, m, rep)
    if family == "row":
        return Sketch(n, m, family, indices=rng.permutation(n)[:m])
    if family == "dense_gaussian":
        # Literal iid N(0,1/m): no pathwise trace normalization.
        return Sketch(n, m, family, matrix=rng.standard_normal((m, n))/np.sqrt(m))
    if family == "weighted_haar":
        Q, R = qr(rng.standard_normal((n, m)), mode="economic", check_finite=False)
        Q *= np.where(np.diag(R) >= 0., 1., -1.)
        weights = np.ones(m)
        weights[1::2] = 2.
        diagonal = n/np.dot(weights, weights)*weights**2
        return Sketch(n, m, family, matrix=np.sqrt(diagonal)[:, None]*Q.T,
                      diagonal=diagonal)
    raise ValueError(f"Unknown sketch: {family}")


def fit_pair(X, Y, covariance, L, beta, S):
    """Two original responses, one actual compressed-design QR factorization."""
    n, d = X.shape
    m = S.m
    B, C = S.apply(X), S.apply(Y)
    Q, R = qr(B, mode="economic", check_finite=False)
    if not np.isfinite(R).all() or np.any(np.abs(np.diag(R)) <= np.finfo(float).tiny):
        raise NumericalFailure("Nonfinite or numerically singular QR factor")
    estimate = solve_triangular(R, Q.T @ C, check_finite=False)
    U = Y - X @ estimate
    SU, AU = S.apply(U), None
    AU = S.adjoint(SU)
    RSS = np.sum(U*U, axis=0)
    inverse_r = solve_triangular(R, np.eye(d), check_finite=False)
    sensitivity_factor = L.T @ inverse_r
    t = float(np.sum(sensitivity_factor*sensitivity_factor))
    if S.family == "row":
        b = n/m*(m-d)
        projected = np.sum(SU*SU, axis=0)/(n/m)
    elif S.diagonal is not None:
        b = float(np.sum(S.diagonal)-np.sum(Q*Q*S.diagonal[:, None]))
        projected = np.sum(SU*SU/S.diagonal[:, None], axis=0)
    else:
        SQ = S.matrix.T @ Q
        b = float(np.sum(S.matrix*S.matrix)-np.sum(SQ*SQ))
        if m < n:
            D = S.matrix @ S.matrix.T
            Dc = cho_factor(D, lower=True, check_finite=False)
            projected = np.sum(SU*cho_solve(Dc, SU, check_finite=False), axis=0)
        else:
            projected = np.full(2, np.nan)
    if not np.isfinite(b) or b <= 0. or not np.isfinite(t) or t <= 0.:
        raise NumericalFailure(f"Invalid b_A={b} or sensitivity trace={t}")
    coefficient_error = L.T @ (estimate-beta[:, None])
    risk = np.array([1., 1./3.]) + np.sum(coefficient_error**2, axis=0)
    obs = np.sum((U+d/b*AU)**2, axis=0)/n
    sens = np.sum((U+t*AU)**2, axis=0)/n
    gcv = RSS/n/(1-d/n)**2
    bellec = RSS/(n-2*d+d*d/m)  # Published row formula; explicitly exploratory for other sketches.
    if m < n:
        complement = RSS-projected
        tolerance = 100*np.finfo(float).eps*np.maximum.reduce((np.ones(2), RSS, projected))
        if np.any(complement < -tolerance):
            raise NumericalFailure(f"Negative complementary sum of squares: {complement}")
        perp = np.maximum(0., complement)/(n-m)
    else:
        perp = np.full(2, np.nan)
    compressed_residual = C-B @ estimate
    denominator = np.linalg.norm(B)*np.linalg.norm(C, axis=0)
    normal_error = np.linalg.norm(B.T @ compressed_residual, axis=0)/np.maximum(
        denominator, np.finfo(float).tiny)
    values = np.stack((risk, obs, sens, gcv, bellec, perp))
    required = values[:4]
    if (not np.isfinite(required).all() or not np.isfinite(normal_error).all()
            or np.any(normal_error > 1.e-9)):
        raise NumericalFailure("Nonfinite criterion or failed normal-equation diagnostic")
    if not np.isfinite(bellec).all():
        raise NumericalFailure("Nonfinite Bellec formula or exploratory extension")
    if m < n and not np.isfinite(perp).all():
        raise NumericalFailure("Nonfinite complementary estimate")
    return values, dict(b_A=b, t_sens=t, RSS=RSS, normal_equation_error=normal_error)


def base_record(cell, rep, seed, family, noise):
    n, d, m = cell["n"], cell["d"], cell["m"]
    design = f"{STUDY}:{seed}:X:n{n}:d{d}:rep{rep}"
    noise_id = f"{STUDY}:{seed}:eps:{noise}:n{n}:d{d}:rep{rep}"
    sketch_id = f"{STUDY}:{seed}:S:{family}:n{n}:d{d}:m{m}:rep{rep}"
    return dict(cell, rep=rep, seed=seed, sketch=family, noise=noise,
                sigma2=1. if noise == "gaussian" else 1./3.,
                design_draw_id=design, noise_draw_id=noise_id, sketch_draw_id=sketch_id,
                fit_draw_id=f"{sketch_id}:{noise}",
                reason_R_B="", RB_scope="published_row_formula" if family == "row" else "exploratory_extension",
                reason_R_perp="" if m < n else "m_equals_n")


def one_rep(cell, rep, seed):
    rows = []
    try:
        data = generate_data(cell["n"], cell["d"], rep, seed)
        data_error = None
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        data_error = exc
    for family in SKETCHES:
        try:
            if data_error is not None:
                raise NumericalFailure(str(data_error))
            S = make_sketch(cell["n"], cell["d"], cell["m"], rep, seed, family)
            values, diagnostics = fit_pair(*data, S)
            for j, noise in enumerate(NOISES):
                row = base_record(cell, rep, seed, family, noise)
                row.update(dict(zip(VALUES, values[:, j])), status="ok", failure_message="")
                row.update({key: float(value[j]) if isinstance(value, np.ndarray) else value
                            for key, value in diagnostics.items()})
                rows.append(row)
        except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            for noise in NOISES:
                row = base_record(cell, rep, seed, family, noise)
                row.update({key: np.nan for key in VALUES+(
                    "b_A", "t_sens", "RSS", "normal_equation_error")})
                row.update(status="numerical_failure", failure_message=str(exc))
                rows.append(row)
    return rows


def environment():
    return dict(python=sys.version, platform=platform.platform(), host=platform.node(),
                packages={name: importlib.metadata.version(name) for name in
                          ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl")},
                threadpools=threadpool_info(), blas_threads_per_worker=1)


def specification(seed, reps):
    return dict(schema=SCHEMA, study=STUDY, seed=seed, repetitions_per_setting=reps,
                production=(reps == REPS and seed == SEED), cells=cells(), sketches=list(SKETCHES),
                noises=list(NOISES), sigma2=dict(zip(NOISES, [1., 1./3.])),
                population={"covariance": "AR(1)", "correlation": .5,
                            "signal_variance": 4., "signal_direction": "first min(10,d) equal"},
                bellec_scope="published row formula; same formula explicitly exploratory for weighted Haar/dense Gaussian",
                sketch_rules={"row": "sqrt(n/m) times selected rows",
                              "weighted_haar": "Haar rows, alternating weights 1,2; trace(S.T S)=n",
                              "dense_gaussian": "iid N(0,1/m); no pathwise normalization"})


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("."+path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def check_manifest(manifest, seed=None, reps=None, require_source=False):
    spec = specification(manifest["specification"]["seed"] if seed is None else seed,
                         manifest["specification"]["repetitions_per_setting"] if reps is None else reps)
    if manifest.get("specification") != spec:
        raise ValueError("Manifest does not match the requested seed, repetitions, or exact grid")
    if require_source and manifest["source_sha256"] != digest(__file__):
        raise ValueError("Resume requires the exact source SHA256 recorded in the manifest")
    if require_source and manifest["environment"]["packages"] != environment()["packages"]:
        raise ValueError("Resume requires matching scientific package versions")
    return spec


def cell_folder(out, cell):
    return Path(out)/"cells"/f"cell-{cell['cell_index']:03d}-n{cell['n']:04d}-d{cell['d']:04d}-m{cell['m']:04d}"


def validate_cell(frame, cell, seed, reps_done):
    if len(frame) != 6*reps_done:
        raise ValueError(f"Incorrect record count in cell {cell['cell_index']}")
    expected = {(rep, family, noise) for rep in range(reps_done)
                for family in SKETCHES for noise in NOISES}
    actual = set(frame[["rep", "sketch", "noise"]].itertuples(index=False, name=None))
    if actual != expected or frame.fit_draw_id.duplicated().any():
        raise ValueError("Missing, duplicate, or unexpected replicate/sketch/noise IDs")
    for key, value in dict(cell, seed=seed).items():
        if not frame[key].eq(value).all():
            raise ValueError(f"Inconsistent cell field {key}")
    if not frame.status.isin(["ok", "numerical_failure"]).all():
        raise ValueError("Unknown fit status")
    for row in frame.itertuples(index=False):
        expected_base = base_record(cell, int(row.rep), seed, row.sketch, row.noise)
        for key in ("design_draw_id", "noise_draw_id", "sketch_draw_id", "fit_draw_id",
                    "reason_R_B", "RB_scope", "reason_R_perp", "sigma2"):
            if getattr(row, key) != expected_base[key]:
                raise ValueError(f"Invalid identity or availability field {key}")
    ok = frame.status.eq("ok")
    for key in VALUES+("b_A", "t_sens", "RSS", "normal_equation_error"):
        available = ok.copy()
        if key == "R_perp":
            available &= frame.m.lt(frame.n)
        if not np.isfinite(frame.loc[available, key]).all():
            raise ValueError(f"Nonfinite available value: {key}")
        if not frame.loc[~available, key].isna().all():
            raise ValueError(f"Unexpected value outside availability: {key}")
    if ((frame.loc[ok, ["b_A", "t_sens"]] <= 0.).any().any()
            or (frame.loc[ok, "normal_equation_error"] > 1.e-9).any()):
        raise ValueError("Invalid calibration mass, trace, or normal-equation diagnostic")
    for key in ("RSS", "normal_equation_error")+VALUES:
        finite = frame.loc[ok, key].dropna()
        if finite.lt(0.).any():
            raise ValueError(f"Negative norm or risk value: {key}")
    if frame.loc[ok, "risk"].lt(frame.loc[ok, "sigma2"]).any():
        raise ValueError("Actual risk below its noise variance")
    if frame.loc[~ok, "failure_message"].eq("").any():
        raise ValueError("Numerical failure without reason")


def load_checkpoint(out, cell, seed):
    folder = cell_folder(out, cell)
    candidates = sorted(folder.glob("checkpoint-*.json")) if folder.exists() else []
    if not candidates:
        return pd.DataFrame(), 0
    meta = json.loads(candidates[-1].read_text())
    source = folder/meta["csv"]
    if meta["cell"] != cell or meta["seed"] != seed or digest(source) != meta["sha256"]:
        raise ValueError("Checkpoint metadata, identity, or content hash mismatch")
    frame = pd.read_csv(source, keep_default_na=False, na_values=["NaN"], float_precision="round_trip")
    validate_cell(frame, cell, seed, meta["repetitions_completed"])
    return frame, meta["repetitions_completed"]


def write_checkpoint(out, cell, seed, frame, reps_done):
    validate_cell(frame, cell, seed, reps_done)
    folder = cell_folder(out, cell)
    folder.mkdir(parents=True, exist_ok=True)
    source = folder/f"checkpoint-{reps_done:06d}.csv.gz"
    temporary = source.with_name("."+source.name+".tmp")
    frame.to_csv(temporary, index=False, na_rep="NaN", compression="gzip", float_format="%.17g")
    temporary.replace(source)
    meta = dict(cell=cell, seed=seed, csv=source.name, sha256=digest(source),
                repetitions_completed=reps_done, records=len(frame),
                numerical_failures=int(frame.status.ne("ok").sum()))
    metadata = source.with_name(source.name.removesuffix(".csv.gz")+".json")
    atomic_json(metadata, meta)
    # Keep the old checkpoint until both new data and metadata are committed.
    for older in folder.glob("checkpoint-*"):
        if older not in (source, metadata):
            older.unlink()


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return (float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values)))) if len(values)>1 else (
        (float(values[0]), np.nan) if len(values) else (np.nan, np.nan))


def summarize(frame):
    rows = []
    for key, group in frame.groupby(list(KEYS), sort=True):
        base = dict(zip(KEYS, key))
        for method in VALUES:
            pair = group[["risk", method]].to_numpy() if method != "risk" else np.column_stack((group.risk, group.risk))
            available = np.isfinite(pair).all(axis=1)
            values, truth = pair[available, 1], pair[available, 0]
            mean, se = mean_mcse(values)
            bias, bse = mean_mcse(values-truth)
            mae, mase = mean_mcse(np.abs(values-truth))
            mse, msse = mean_mcse((values-truth)**2)
            rmse = float(np.sqrt(mse))
            rmse_se = msse/(2*rmse) if rmse > 0. else (0. if rmse == 0. else np.nan)
            reason = "m_equals_n" if method == "R_perp" and base["m"] == base["n"] else ""
            scope = ("published_row_formula" if base["sketch"] == "row" else "exploratory_extension") if method == "R_B" else ""
            rows.append(dict(base, method=method, attempted=len(group), count=int(available.sum()),
                unavailable=int((~available).sum()), availability_reason=reason, RB_scope=scope,
                mean=mean, mean_mcse=se, mean_ci_low=mean-1.96*se, mean_ci_high=mean+1.96*se,
                paired_bias=bias, paired_bias_mcse=bse,
                paired_bias_ci_low=bias-1.96*bse, paired_bias_ci_high=bias+1.96*bse,
                mae=mae, mae_mcse=mase, mse=mse, mse_mcse=msse,
                rmse=rmse, rmse_mcse=rmse_se, paired_rmse=rmse, paired_rmse_mcse=rmse_se))
    return pd.DataFrame(rows)


def write_table(summary, out):
    names = dict(row="Row", weighted_haar="WH", dense_gaussian="DG")
    headers = (r"$R$", r"$\widehat R_o$", r"$\widehat R_u$",
               r"$\widehat R_{\rm GCV}$", r"$\widehat R_B$", r"$\widehat R_c$")
    for mean_field, se_field, filename in (("mean", "mean_mcse", "risk_means_n1000.tex"),
            ("paired_bias", "paired_bias_mcse", "bias_n1000.tex"),
            ("rmse", "rmse_mcse", "rmse_n1000.tex")):
        methods = VALUES if mean_field == "mean" else VALUES[1:]
        titles = headers if mean_field == "mean" else headers[1:]
        lines = [r"\begin{tabular}{@{}lll"+"r"*len(methods)+r"@{}}", r"\toprule",
                 r"Sketch & Errors & $(d/n,m/n)$ & "+" & ".join(titles)+r" \\", r"\midrule"]
        for i, (gamma, rho) in enumerate(RATIOS):
            if i:
                lines.append(r"\midrule")
            for family in SKETCHES:
                for noise in NOISES:
                    group = summary[(summary.n == 1000) & (summary.gamma == gamma) &
                        (summary.rho == rho) & (summary.sketch == family) & (summary.noise == noise)]
                    entries = []
                    for method in methods:
                        row = group[group.method == method].iloc[0]
                        entry = "--" if row.availability_reason == "m_equals_n" else f"{row[mean_field]:.3f} ({row[se_field]:.3f})"
                        entries.append(entry)
                    lines.append(names[family]+" & "+("G" if noise == "gaussian" else "U")+
                        f" & $({gamma:.1f},{rho:.1f})$ & "+" & ".join(entries)+r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", f"% Metric: {mean_field} (Monte Carlo SE), n=1000.",
                  "% Row/WH/DG: row sampling/weighted Haar/dense Gaussian; G/U: Gaussian/Uniform[-1,1].",
                  "% R_B uses Bellec's formula for all sketch families; --: m=n, no complementary directions.", ""]
        (Path(out)/filename).write_text("\n".join(lines))


def make_figures(summary, out, spec):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = (r"True $R$", r"$\widehat R_o$", r"$\widehat R_u$",
              r"$\widehat R_{\mathrm{GCV}}$", r"$\widehat R_B$", r"$\widehat R_c$")
    colors = ("black", "#0072B2", "#009E73", "#D55E00", "#7B3294", "#CC79A7")
    markers = ("o", "^", "s", "D", "x", "+")
    styles = ("-", "--", "-.", ":", (0, (6, 2, 1, 2)), (0, (2, 2)))
    folder = Path(out)/"figures"
    folder.mkdir(exist_ok=True)
    with PdfPages(folder/"risk_curves.pdf") as combined:
        for noise in NOISES:
            fig, axes = plt.subplots(3, 4, figsize=(7.5, 6.7), sharex=True, sharey=False)
            for i, family in enumerate(SKETCHES):
                for j, (gamma, rho) in enumerate(RATIOS):
                    ax = axes[i, j]
                    group = summary[(summary.noise == noise) & (summary.sketch == family) &
                                    (summary.gamma == gamma) & (summary.rho == rho)]
                    for k, method in enumerate(VALUES):
                        values = group[(group.method == method) & (group["count"] > 0)].sort_values("n")
                        if values.empty:
                            continue
                        x = values.n.to_numpy()
                        ax.fill_between(x, values.mean_ci_low.to_numpy(), values.mean_ci_high.to_numpy(),
                                        color=colors[k], alpha=.065, linewidth=0)
                        ax.plot(x, values["mean"].to_numpy(), label=labels[k], color=colors[k],
                                linestyle=styles[k], marker=markers[k], markevery=4, markersize=3.,
                                markerfacecolor=colors[k] if k == 0 else "none", linewidth=1.25 if k == 0 else 1.05)
                    ax.grid(alpha=.18)
                    ax.set_xlim(100, 1000)
                    ax.set_xticks([100, 400, 700, 1000])
                    if i == 0:
                        ax.set_title(rf"$(d/n,m/n)=({gamma:.1f},{rho:.1f})$")
                    if j == 0:
                        ax.set_ylabel({"row": "Row sampling", "weighted_haar": "Weighted Haar",
                                       "dense_gaussian": "Dense Gaussian"}[family]+"\nMean risk")
                    if i == 2:
                        ax.set_xlabel(r"Sample size $n$")
                        tick_labels = ax.get_xticklabels()
                        tick_labels[0].set_horizontalalignment("left")
                        tick_labels[-1].set_horizontalalignment("right")
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            legend_panel = summary[(summary.noise == noise) & (summary.sketch == SKETCHES[0]) &
                                   (summary.gamma == RATIOS[0][0]) & (summary.rho == RATIOS[0][1])]
            lower = float(legend_panel.mean_ci_low.min())
            upper = float(legend_panel.mean_ci_high.max())
            span = upper - lower
            axes[0, 0].set_ylim(lower - .06 * span, upper + .90 * span)
            axes[0, 0].legend(handles, legend_labels, loc="upper left", ncol=2,
                              fontsize=6.5, framealpha=.97, borderpad=.35,
                              handlelength=1.4, handletextpad=.35,
                              columnspacing=.65, labelspacing=.45)
            title = "Gaussian errors" if noise == "gaussian" else "Uniform errors"
            suffix = "" if spec["production"] else " — validation run"
            fig.suptitle(title+suffix, y=.98)
            fig.subplots_adjust(left=.108, right=.978, top=.925, bottom=.075, hspace=.27, wspace=.19)
            stem = "risk_curves_gaussian" if noise == "gaussian" else "risk_curves_uniform"
            fig.savefig(folder/(stem+".pdf"))
            fig.savefig(folder/(stem+".png"), dpi=220)
            combined.savefig(fig)
            plt.close(fig)


def validate_raw(folder, require_paper=True):
    """Validate the complete cell archive; return checked specification and counts.

    Expected applicability NaNs are accepted. Failed, incomplete, duplicate or
    altered records are rejected. require_paper also requires exactly 1000 reps.
    """
    out = Path(folder)
    manifest = json.loads((out/"manifest.json").read_text())
    spec = check_manifest(manifest)
    if require_paper and not spec["production"]:
        raise ValueError("A paper run requires seed 2026100901 and exactly 1000 repetitions per setting")
    expected = {cell_folder(out, c) for c in cells()}
    actual = set((out/"cells").iterdir()) if (out/"cells").exists() else set()
    if actual != expected:
        raise ValueError("The archive must contain exactly the 76 expected cell directories")
    total, csvs = 0, []
    for cell in cells():
        frame, done = load_checkpoint(out, cell, spec["seed"])
        if done != spec["repetitions_per_setting"]:
            raise ValueError(f"Incomplete cell {cell['cell_index']}: {done} repetitions")
        if frame.status.ne("ok").any():
            raise ValueError(f"Cell {cell['cell_index']} contains numerical failures")
        total += len(frame)
        csvs.append(str(cell_folder(out, cell)/f"checkpoint-{done:06d}.csv.gz"))
    if total != 456*spec["repetitions_per_setting"]:
        raise ValueError("Incorrect total fitted-model count")
    counts = dict(settings=456, repetitions_per_setting=spec["repetitions_per_setting"],
                  fits_attempted=total, paired_factorizations_attempted=total//2,
                  numerical_failures=0, fits_successful=total, complete_for_reporting=True)
    if manifest.get("raw_sha256") is not None and digest(out/"raw.csv.gz") != manifest["raw_sha256"]:
        raise ValueError("Aggregate raw.csv.gz content hash disagrees with the manifest")
    if manifest.get("counts") is not None and manifest["counts"] != counts:
        raise ValueError("Final manifest counts disagree with the validated cell archive")
    return dict(specification=spec, counts=counts, cell_csvs=csvs)


def render(out):
    out = Path(out)
    manifest = json.loads((out/"manifest.json").read_text())
    spec = check_manifest(manifest)
    seed, reps = spec["seed"], spec["repetitions_per_setting"]
    expected_folders = {cell_folder(out, c) for c in cells()}
    actual_folders = set((out/"cells").iterdir()) if (out/"cells").exists() else set()
    if actual_folders != expected_folders:
        raise ValueError("The archive must contain exactly the 76 expected cell directories")
    summaries, total, failures = [], 0, []
    aggregate = out/".raw.csv.gz.tmp"
    with gzip.open(aggregate, "wt") as stream:
        for cell in cells():
            frame, done = load_checkpoint(out, cell, seed)
            if done != reps:
                raise ValueError(f"Incomplete cell {cell['cell_index']}: {done}/{reps}")
            frame.to_csv(stream, index=False, header=(total == 0), na_rep="NaN", float_format="%.17g")
            total += len(frame)
            failures.append(frame[frame.status.ne("ok")])
            summaries.append(summarize(frame))
    if total != 76*6*reps:
        raise ValueError("Incorrect total count")
    aggregate.replace(out/"raw.csv.gz")
    failure_frame = pd.concat(failures, ignore_index=True)
    failure_frame.to_csv(out/"failures.csv", index=False, na_rep="NaN")
    counts = dict(settings=456, repetitions_per_setting=reps, fits_attempted=total,
                  paired_factorizations_attempted=76*3*reps, numerical_failures=len(failure_frame),
                  fits_successful=total-len(failure_frame), complete_for_reporting=len(failure_frame) == 0)
    manifest.update(counts=counts, raw_sha256=digest(out/"raw.csv.gz"),
                    status="numerical_failure" if len(failure_frame) else "complete")
    atomic_json(out/"manifest.json", manifest)
    if len(failure_frame):
        raise RuntimeError("Numerical failures retained in cell archives/raw.csv.gz/failures.csv; no report generated")
    validate_raw(out, require_paper=spec["production"])
    atomic_json(out/"metadata.json", manifest)
    summary = pd.concat(summaries, ignore_index=True)
    if len(summary) != 456*6 or summary[list(KEYS)+["method"]].duplicated().any():
        raise ValueError("Incorrect or duplicate summary grid")
    summary.to_csv(out/"summary.csv", index=False, na_rep="NaN", float_format="%.17g")
    write_table(summary, out)
    make_figures(summary, out, spec)
    readme = f"""# Original-data mean-risk experiment

{'PRODUCTION' if spec['production'] else 'VALIDATION / NONPRODUCTION'}: seed {seed}; {reps:,} independent repetitions per setting.
The fixed grid crosses n=100,150,...,1000, four dimension ratios
(0.3,0.5), (0.5,0.7), (0.7,0.9), (0.8,1.0), three sketch families,
and Gaussian / literal Uniform[-1,1] errors: 456 settings and {total:,} fitted models.
The design is Gaussian AR(1), correlation 0.5; beta is fixed, sparse, and beta' Sigma beta=4.
Gaussian noise variance is 1; Uniform[-1,1] variance is 1/3, including in the exact risk target.

Each replication generates the actual X and two y vectors. Sketch families share X and both
errors; the errors share each sketch and compressed-design QR factorization.
There is no risk-law sampling, rotated-error generation, ridge stabilization, truncation,
replacement draw, or trimming. The exact realized risk is sigma2+(beta_hat-beta)'Sigma(beta_hat-beta).
The AR(1) recurrence produces the same feature matrix as multiplying the same Gaussian
innovations by the population Cholesky factor, up to floating-point rounding.

Row sampling uses sqrt(n/m); weighted Haar uses alternating weights 1 and 2 normalized
to trace(S' S)=n; dense Gaussian entries are independent N(0,1/m), without pathwise
normalization. R_B=RSS/(n-2d+d^2/m) is the published comparator for literal row sampling,
using the existing manuscript's Bellec et al. comparison formula. We use this formula
regardless of sketch family and denote it by R_B throughout. Its use for weighted Haar
and dense Gaussian sketches is an empirical comparison; no validity theorem from the
published row-sampling result is asserted for those sketches. RB_scope retains this
applicability distinction in the raw records. R_perp uses the actual orthogonal
complement residual for both error families when m<n, and is unavailable when m=n
(code --). No exact Gaussian conditional-calibration claim is made for Uniform errors.

summary.csv gives empirical means, Monte Carlo SEs and pointwise 95% intervals
(mean +/- 1.96 SE), and within-fit paired bias relative to exact risk. MAE and MSE
are auxiliary columns only. The figures plot means of risk and estimates on linear
axes; bands are pointwise Monte Carlo intervals. No simultaneous coverage is claimed.
The six criteria share each fit; families share datasets. The mean, paired-bias, and RMSE tables report n=1000
means (Monte Carlo SE), with explicit availability codes.

The cells/ directories contain gzip CSV checkpoints and hash-verified metadata.
A new checkpoint is committed every {CHECKPOINT} complete replications, before removing
the preceding checkpoint. raw.csv.gz aggregates all checked complete cells.
Failures are retained without replacement and prevent publication of summary figures.
manifest.json records the exact specification, source hash, environment and counts.
Resume requires explicit matching seed/repetition options, the same source SHA256,
and matching scientific package versions; changing worker count preserves all streams.

Commands: python risk_curves.py run --output DIR --jobs 6;
python risk_curves.py resume --output DIR --jobs 6;
python risk_curves.py render --output DIR.
For smoke validation use --reps 20 on both run and resume; that output is labeled nonproduction.
The paper designation requires both the default seed 2026100901 and exactly 1000 repetitions.
"""
    (out/"README.md").write_text(readme)
    print(f"Validated {total:,} fits and wrote mean-risk figures to {out}", flush=True)


def run(args, resume=False):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/"manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        check_manifest(manifest, args.seed, args.reps, require_source=True)
    else:
        if manifest_path.exists() or any(out.iterdir()):
            raise FileExistsError("Run requires an empty output directory; use explicit resume")
        manifest = dict(specification=specification(args.seed, args.reps), source_sha256=digest(__file__),
                        environment=environment(), created_utc=datetime.now(timezone.utc).isoformat(),
                        status="running", worker_history=[])
    manifest.setdefault("worker_history", []).append(dict(jobs=args.jobs,
        started_utc=datetime.now(timezone.utc).isoformat(), action="resume" if resume else "run"))
    atomic_json(manifest_path, manifest)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for cell in cells():
            prior, done = load_checkpoint(out, cell, args.seed)
            if done > args.reps:
                raise ValueError("Checkpoint exceeds requested repetition count")
            if done == args.reps:
                print(f"Cell {cell['cell_index']+1}/76 verified complete", flush=True)
                continue
            rows = prior.to_dict("records") if done else []
            worker = partial(one_rep, cell, seed=args.seed)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for count, result in enumerate(pool.map(worker, range(done, args.reps)), done+1):
                    rows.extend(result)
                    if count % CHECKPOINT == 0 or count == args.reps:
                        write_checkpoint(out, cell, args.seed, pd.DataFrame(rows), count)
                        print(f"Cell {cell['cell_index']+1}/76 n={cell['n']} d={cell['d']} m={cell['m']}: "
                              f"{count}/{args.reps} reps; elapsed {time.perf_counter()-start:.1f}s", flush=True)
            print(f"Cell {cell['cell_index']+1}/76 complete", flush=True)
    manifest = json.loads(manifest_path.read_text())
    manifest["worker_history"][-1]["elapsed_seconds"] = time.perf_counter()-start
    atomic_json(manifest_path, manifest)
    render(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume", "render"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        if name != "render":
            sub.add_argument("--reps", type=int, default=REPS, help="1000 production; any other value is a smoke run")
            sub.add_argument("--seed", type=int, default=SEED)
            sub.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.command == "render":
        render(args.output)
    else:
        if args.reps < 2 or args.jobs < 1:
            parser.error("--reps must be at least 2 and --jobs positive")
        run(args, resume=args.command == "resume")


if __name__ == "__main__":
    main()
