# Reproduction code for *Prediction Risk After Observation Compression*

It generates the current simulation figures and numerical tables in Section 5
and Appendix A. The CPAL manuscript keeps Figures 1–3 in the main text and Figures 4–6 plus all three
simulation tables in Appendix A. Section 5 contains the study protocols;
Appendix A contains paired bias/RMSE definitions and standard errors,
the weighted-Haar/dense-Gaussian nested-sketch protocol, and additional
findings and displays. Section 5.1 now uses the completed Unity rerun at
`n=300`, `m=150` and `d in {100,300,500}`.
The simulation studies use simulated data and require no external dataset
or LaTeX installation. The expanded Section 5.2 study uses actual original-data fits and
plots mean risks, including the exact risk of each fitted predictor.

## Install

Use Python 3.9 or newer. From the directory containing `Code_CPAL`:

```bash
cd Code_CPAL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

These commands use a Linux/macOS shell. `environment-paper.txt` records the
direct package versions used in the original Python 3.13.13 runs; it can be
used instead of `requirements.txt` in a compatible Python environment.
`environment-risk-curves.txt` records the direct versions used for the new
Section 5.2 run, tested with Python 3.13.5. Its run manifest also records the
BLAS implementation and operating system.
The current Section 5.3 rerun uses the same local package versions. Both
run metadata files record the environment and BLAS implementation in
`reports/mse-1000-20261012/results/` at the repository root.

## Reproduce all simulations

```bash
python reproduce.py run --jobs 6
```

Results are written to `runs/paper/<study>/`, including raw draws, summaries,
PDF/PNG figures, LaTeX table fragments, and run metadata. The studies run
sequentially, using up to six workers within each study. Lower `--jobs` to suit
your machine.

The runner fixes the current manuscript's parameter grids and seeds.
Section 5.2 uses seed `2026100901` and 1,000 repetitions per setting. The
conditional study uses 100 independently frozen fits per setting, each with
1,000 fresh complementary-data draws. Both threshold studies use 1,000
repetitions. The conditional grid is `n=300`, `m=150`, `d in {100,300,500}`,
with row and dense Gaussian sketches (six settings).
Conditional and both threshold-study seeds are `2026092601`. The runner
includes the following four studies.

| Study (`--study`) | Manuscript outputs |
|---|---|
| `conditional` | Section 5.1, Figure 1 |
| `risk_curves` | Section 5.2, Figure 2: Gaussian risk curves; Appendix A.1, Figure 4 and Tables 1–3: uniform curves, n=1000 means and paired bias/RMSE |
| `threshold` | Section 5.3, Figure 3; Appendix A.2: additional row-sketch summaries |
| `threshold_shapes` | Section 5.3: shared grid and intervals; Appendix A.2, Figures 5–6: weighted Haar/dense Gaussian nested-sketch protocol and MSE comparisons |

Tables 1–3 contain simulation means, paired bias, and RMSE. To run a single study or choose a fresh destination:

```bash
python reproduce.py run --study risk_curves --jobs 6 --output-root runs/risk-rerun
```

Existing nonempty study directories are rejected. Failed draws are retained
without replacement; incomplete runs are rejected. Conditional runs save
atomic per-fit NPZ, raw CSV and diagnostic JSON checkpoints; interrupted
outputs are retained, and reruns require a fresh output root. Conditional
completion checks independently validate all draw identities, raw-data
algebra, frozen models, checkpoints and summaries. The risk-curve and threshold-shapes studies additionally support explicit resume,
as described below.

## Completed runs and retained records

The completed manuscript runs had no numerical failures, replacement draws,
or trimming. Their retained outputs are at the following paths, relative
to the repository root:

| Study | Completed work | Local archive |
|---|---|---|
| `conditional` | 600 frozen fits and 600,000 complementary-data draws | `reports/conditional-20261012/results/` |
| `risk_curves` | 456,000 fitted-model evaluations; 228,000 compressed-design factorizations | `reports/expanded-risk-20261009/` |
| `threshold` | 45,000 fits from 9,000 training datasets | `reports/mse-1000-20261012/results/threshold/` |
| `threshold_shapes` | 90,000 additional fits using the same 9,000 training datasets | `reports/mse-1000-20261012/results/threshold_shapes/` |

The two Section 5.3 studies jointly contain 135,000 fits, with each dataset
paired across all three sketch families and five nested sketch sizes.
The updated studies fit actual compressed training data; no fitted-risk
distribution or chi-square pivot is used to generate their results.

The three updated archives retain raw fit records, summaries, frozen
generating sources, package versions, computing metadata, and independent
numerical validation. The Section 5.2 raw records include risk estimates,
true risks, residual sums of squares, calibration masses, sensitivity
traces, scaled normal-equation diagnostics, and comparator-scope fields.
Its dimension-cell checkpoints are hash-verified. The weighted Haar/dense
Gaussian extension also retains hash-verified checkpoints. Run metadata
records the exact grid, seeds, repetitions, generating source hashes, and
computing environment. Validators check dimensions and draw identities,
then recompute summaries and paired intervals from the raw records before
rendering. Resume requirements and study-specific commands appear below.

The current conditional-calibration study retains seed `2026092601` and uses
Unity simulation job `11263874` and independent validation job `11263884`.
Its six settings use `n=300`, `m=150`, `d in {100,300,500}`, 100 frozen fits
each and 1,000 fresh complementary-data blocks per fit. The fit is least
squares at `d=100` and minimum norm at `d=300` and `d=500`; `d=n` does not
equal the compressed interpolation boundary `d=m`. Stream keys contain the
version `n300-m150-v1`, dimensions, family and fit index, with separate sketch,
retained-data and complementary-data phases. Outer-fit Monte Carlo intervals
use Student t with 99 degrees of freedom; the exact risk interval uses
chi-square quantiles with 150 degrees of freedom. These fitted models are
separate from those in Section 5.2 and 5.3.
The new archive contains all 600 frozen NPZ models, 600,000 draw-level rows,
per-fit checkpoints, summaries, source snapshots, job logs and provenance.
Its independent validation recomputes all metrics and intervals, checks all
artifact hashes, and replays the first 25 complementary blocks of every fit
(15,000 blocks) from the specified Gaussian data generator. All draws and
fits succeeded. The standalone validation command, from this directory, is:

```bash
python validate_conditional.py --report runs/paper/conditional --paper
```

The original Unity validator and its evidence are retained unchanged in the
new report. The bundled validator keeps row-sketch replay and file hashes
exact; Gaussian-sketch replay allows relative error at most four float64
machine epsilons, with zero absolute tolerance, for cross-platform rounding.
Its returned evidence records the local environment and replay discrepancy.

## Simulation figure presentation

All six current simulation figures use one shared legend inside the
upper-left panel. Figures 2 and 4 have titles "Gaussian errors" and
"Uniform errors"; Figure 3's guide labels are `rho_*(gamma)` and
`rho_n^{\#}`. The current notation uses $\widehat R_o$,
$\widehat R_u$, and $\widehat R_c$ in risk figures, MSE axes, and tables.
The manuscript also uses $\widehat\beta_{m,S}$ for the minimum-norm fit.
The subscripts $m$, $u$, $o$, and $c$ stand for minimum-norm, unobservable,
observable, and complementary. The row-whitened sketch is
$F=(SS^\top)^{-1/2}S$, with $P_S=F^\top F$ and complementary projector
$Q_S=I_n-P_S$. The matrix $F$ has orthonormal rows; $Q_S$ acts on the
original observation space. Existing computational basis names and saved
record schemas are retained.

To regenerate current Figures 1–6 and Tables 1–3 from the completed
archives, without rerunning the simulations, use the current renderer.
The saved archives have different directory layouts; temporary aliases
provide the `<input-root>/<study>` layout expected by `reproduce.py`.
From the repository root, run:

```bash
render_workspace="$(mktemp -d /tmp/randomsketch-current-render.XXXXXX)"
mkdir "$render_workspace/input"
ln -s "$PWD/reports/conditional-20261012/results" "$render_workspace/input/conditional"
ln -s "$PWD/reports/expanded-risk-20261009" "$render_workspace/input/risk_curves"
ln -s "$PWD/reports/mse-1000-20261012/results/threshold" "$render_workspace/input/threshold"
ln -s "$PWD/reports/mse-1000-20261012/results/threshold_shapes" "$render_workspace/input/threshold_shapes"

python Code_CPAL/reproduce.py render \
  --input-root "$render_workspace/input" \
  --output-root "$render_workspace/rendered"
```

Outputs are written beneath `$render_workspace/rendered`; the original
records remain unchanged. The MSE inputs above are the current
1,000-repetition runs, and the current sources retain the
`rho_n^{\#}` guide label. For a newly completed run already organized
under `runs/paper/`, use the command in
[Regenerate figures and tables from a completed run](#regenerate-figures-and-tables-from-a-completed-run).

Saved record columns and internal method identifiers remain `R_obs`,
`R_sens`, and `R_perp` for stable schemas; they now display as $\widehat R_o$,
$\widehat R_u$, and $\widehat R_c$. The unobservable risk estimate uses the population covariance and is
unavailable when that covariance is unknown; simulations supply the known
covariance to evaluate it. Labels change no formulas, numerical values,
plotted coordinates, intervals, or styles. Strict checkpoint/resume
checks still require the original generating snapshots described below.

## Expanded mean-risk study

The fixed grid is `n = 100, 150, ..., 1000`, with dimension ratios
`(d/n,m/n) = (0.3,0.5), (0.5,0.7), (0.7,0.9), (0.8,1)`.
It uses row sampling, weighted Haar, and dense Gaussian sketches, crossed
with Gaussian and literal `Uniform[-1,1]` errors. There are 456 settings and
456,000 fitted-model evaluations; two responses share each of 228,000
compressed-design QR factorizations. Sketch families share the original
design and error vectors within each repetition.

Gaussian errors have variance 1; the uniform errors have variance 1/3.
The exact realized response-prediction risk uses the corresponding variance.
Row sketches have scale `sqrt(n/m)`; weighted Haar sketches use alternating
weights 1 and 2, normalized to `trace(S' S) = n`; dense Gaussian entries are
independent `N(0,1/m)` without pathwise trace normalization.

The figures compare true risk, $\widehat R_o$, $\widehat R_u$, GCV, the Bellec
expression, and $\widehat R_c$. We use Bellec's formula regardless of
sketch family and denote
it by `R_B` throughout. Its use for weighted Haar and dense Gaussian sketches
is an empirical comparison; the cited result concerns row sampling.
The complementary estimate $\widehat R_c$ is undefined at `m = n`.
Its uniform-error curves are
empirical comparisons and do not assert the Gaussian conditional pivot.
Mean-risk agreement concerns average calibration; auxiliary paired bias,
MAE, MSE, and RMSE columns describe individual-fit estimation error.

The completed study retains its original generating snapshot in
`reports/expanded-risk-20261009/source/risk_curves.py` at the repository root.
The current renderer uses the common `R_B` notation; the presentation update
preserves all saved numerical results. Archived computations require their
frozen source snapshot for the strict source-hash resume check.

The standalone commands use a single study directory:

```bash
python code/main_studies/risk_curves.py run --output runs/risk-study --jobs 6
python code/main_studies/risk_curves.py resume --output runs/risk-study --jobs 6
python code/main_studies/risk_curves.py render --output runs/risk-study
```

Checkpoint data and hash-verified metadata are committed after every 100
complete repetitions in a dimension cell. Resume requires the same exact
specification, source hash, and package versions. Changing worker count
preserves keyed random streams. No failed draw is redrawn or trimmed.
The renderer validates all 76 dimension cells and their unique identities,
recomputes summaries from raw records, and rejects incomplete/failed runs.
It writes `raw.csv.gz`, `summary.csv`, `manifest.json`, the two individual
risk figures, a combined two-page `figures/risk_curves.pdf`, and three
n=1000 LaTeX table fragments. Each manifest records the generating source
hash and computing environment.

## Updated MSE-comparison studies

Section 5.3 crosses `n = 100, 300, 500` with `d/n = 0.3, 0.5, 0.7`.
At each population size, five row-subsample sizes are the nearest integers
to `n * (3(d/n)/(1 + 2(d/n)) + delta)` for
`delta = -0.08, -0.04, 0, 0.04, 0.08`; exact half-integer ties round upward.
The figure has nine panels labelled by `d/n` and `n`, with actual `m/n`
on the horizontal axis. It compares the paired empirical MSE ratio with
the exact finite-sample ratio and both crossover locations.

The study uses 1,000 repetitions, Gaussian errors with variance one, the
AR(1) covariance and signal used in Section 5.2, and seed `2026092601`.
Its 45 settings contain 45,000 fits from 9,000 datasets. Within each
dataset, the five sketches are nested and both estimators assess the same
fit.
The raw archive validator checks hashes, exact dimensions, all repetition
and draw identities, and every saved summary against the raw fit records.

```bash
python reproduce.py run --study threshold --jobs 6 --output-root runs/threshold-rerun
```

The `threshold_shapes` extension uses the same grid and 1,000 repetitions
for weighted Haar and raw dense Gaussian sketches. Its 90 settings contain
90,000 additional fits from the same 9,000 original datasets as the row
study. At each repetition, Haar-basis prefixes have alternating weights
1 and 2, normalized to `trace(S' S) = n`. Raw standard Gaussian matrix
prefixes are divided by `sqrt(m)`, giving independent `N(0,1/m)` entries
at each sketch size. Both families use nested prefixes across the five sizes.

Each family has a nine-panel empirical MSE-ratio figure with paired
pointwise 95% Monte Carlo intervals and the line at one. The scaled orthogonal
exact MSE curve and crossover formulas are used only in the row figure;
they are not applied to weighted Haar or raw dense Gaussian sketches.
The current row and extension runs are archived at
`reports/mse-1000-20261012/results/threshold/` and
`reports/mse-1000-20261012/results/threshold_shapes/` at the repository root.
They were freshly generated from original data, using 1,000 repetitions in
every setting.

```bash
python reproduce.py run --study threshold_shapes --jobs 6 --output-root runs/shapes-rerun
python code/main_studies/threshold_shapes.py --output runs/shapes-rerun/threshold_shapes --resume --jobs 6
```

The extension checkpoints every 100 complete repetitions in each population
cell. Resume requires identical specifications, generating source hashes,
and scientific package versions. Failed draws are retained without
replacement; a failed or incomplete run cannot be reported as complete.
Use the archived generating sources when resuming an archived computation
after source edits.

## Quick installation check

```bash
python code/main_studies/risk_curves.py run --reps 20 --jobs 1 --output runs/smoke/risk
```

This runs the complete risk-curve grid with 20 repetitions. Its output is
labelled nonproduction and does not reproduce the manuscript's results.
The paper runner accepts only the fixed seed and 1,000 repetitions for this
study.

## Regenerate figures and tables from a completed run

```bash
python reproduce.py render
```

This reads `runs/paper/` and writes to `runs/rendered/` without new simulation
draws. For other locations, supply `--input-root` and `--output-root`.
Risk-curve summaries are recomputed from copied, validated raw checkpoints.
Both threshold studies validate raw fits and recompute the saved summaries
as a check before rendering; the conditional study renders its validated
setting summaries.

The bundle includes `reproduce.py`, shared routines in `code/utils.py` and
`code/main_studies/common.py`, four simulation-study scripts, and the
independent conditional validator. Saved simulation results are not bundled; generate them with
`run` before using `render`. Keyed random streams
preserve seed assignments across worker counts; floating-point results may
differ slightly across package versions and BLAS implementations.

Historical studies and their frozen generating sources are retained separately
and indexed in [reports/README.md](../reports/README.md).
