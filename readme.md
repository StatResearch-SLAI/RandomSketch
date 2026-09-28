# RandomSketch: simulation reproduction

Code and completed simulation results for **Prediction Risk After Observation
Compression**. This repository reproduces every simulation in Section 6 and
Appendix A of the manuscript as of 28 September 2026.

The five studies generate **five figures and five numerical tables**. All data
are simulated; no external dataset, manuscript checkout, or Lean installation
is needed. The numerical study code is copied from the version that generated
the paper's results and final figure layouts.

## Quick start: rebuild the paper figures and tables

Use Python 3.9 or newer and a virtual environment. The commands below use a
Unix shell (Linux or macOS):

```bash
git clone https://github.com/StatResearch-SLAI/RandomSketch.git
cd RandomSketch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py render
```

This reads the completed summaries in `results/` and writes all five figures
as PDF and PNG files, the main accuracy table, and four appendix table
fragments into `runs/rendered/`. It generates no new simulation draws and
leaves the reference results unchanged. Each rendered study records its
input hashes, source hashes, and local plotting environment.

To choose another output directory:

```bash
python reproduce.py render --input-root results --output-root runs/figures-v2
```

LaTeX is not needed to generate the figures or table fragments. To include a
table fragment in a document, load `booktabs` and define the manuscript macros:

```latex
\usepackage{amsmath,booktabs}
\newcommand{\hR}{\widehat{R}}
\newcommand{\GCV}{\mathrm{GCV}}
\newcommand{\bS}{\mathbf{S}}
```

## Rerun all five simulations from the data-generating models

After installing the dependencies:

```bash
python reproduce.py run --jobs 6
```

This generates the original linear-model data, fits the regressions, and
writes new raw draws, summary statistics, diagnostics, figures, tables, and
metadata under `runs/paper/`. It runs the five studies sequentially, with six
workers within each study and one BLAS thread per worker. The command also
generates the four additional accuracy tables. It does not simulate outcomes
from the scalar risk or chi-square laws.

The runner fixes the paper's seeds, grids, and 3,000 repetitions. The
conditional study also fixes 20 fitted summaries per setting. To rerun just
one study, use its name from the table below:

```bash
python reproduce.py run --study accuracy --jobs 6 --output-root runs/accuracy-rerun
```

Existing nonempty study output directories are rejected. Choose a fresh
`--output-root` for a new run; failed draws are retained without replacement
or trimming. A study failure stops the sequential command, and the renderer
rejects incomplete runs. There is no automatic resume or overwrite mode.

All paths passed to the CLI can be absolute. When omitted, default input and
output paths are resolved relative to the repository, independently of the
shell's working directory. Use `python reproduce.py --help` for the command
overview.

## Studies and manuscript outputs

Paths below are relative to `results/` for the archived results and to the
selected output root for a fresh run.

| Study name | Manuscript location | Outputs | Work in the paper run |
|---|---|---|---|
| `conditional` | Section 6.1, Figure 1 | `conditional/figures/complementary_calibration.pdf` | 120 frozen fits; 360,000 complementary-data draws |
| `accuracy` | Section 6.2, Figure 2, Table 2 | `accuracy/figures/accuracy.pdf`, `accuracy/accuracy_table.tex` | 24 settings; 72,000 fit evaluations |
| `threshold` | Section 6.3, Figure 3 | `threshold/figures/threshold.pdf` | 30 settings; 90,000 fit evaluations |
| `accuracy` | Appendix A.1, Tables 3--6 | `accuracy/appendix/{bias,rmse,scaled_mae,paired_mae}_table.tex` | Reuses the Section 6.2 results |
| `dense_benchmarks` | Appendix A.2, Figure 4 | `dense_benchmarks/figures/dense_benchmarks.pdf` | 9 settings; 27,000 fits |
| `isotropic_benchmarks` | Appendix A.3, Figure 5 | `isotropic_benchmarks/figures/isotropic_benchmarks.pdf` | 18 settings; 54,000 fits |

Table 1 summarizes the theoretical regimes and is not a simulation output.
The [study protocol](docs/studies.md) gives the full parameter grids, sketch
scalings, estimands, Monte Carlo interval formulas, and pairing rules.

The three main studies use seed **2026092601**; the two appendix studies use
seed **2026092701**. All nonconditional studies use sample sizes 128, 256,
and 512. Noise variance is one and signal prediction variance is four.
Features have AR(1) covariance with correlation 0.5, except for the isotropic
minimum-norm study, which uses identity covariance as required by its theorem.

Important distinctions when interpreting repetitions:

- Conditional uncertainty is aggregated over **20 independent frozen fits**
  per setting, each with 3,000 fresh complementary-data draws.
- The accuracy and threshold studies share **18,000 central Gaussian
  row-sketch fits**. Their 72,000 and 90,000 fit evaluations are not all
  independent fits.
- Methods evaluated on the same fit are paired. Sketch families can share
  data, and the threshold study uses nested row subsets across sketch sizes.
- The lower isotropic benchmark panels measure RMSE around the deterministic
  limiting risk, separately from estimating each fitted model's own risk.

## Environment and numerical reproducibility

`requirements.txt` lists the six runtime dependencies. `environment-paper.txt`
records their exact versions in the original Unity jobs, which used
Python **3.13.13**. To use those recorded direct-package versions in a suitable
Python environment:

```bash
python -m pip install -r environment-paper.txt
```

The full original `pip freeze` listings are under
`provenance/{main,appendix}/logs/environment-*.txt`; metadata in each result
directory records the operating system and BLAS configuration. The pinned
direct-package list does not install Python itself or reproduce the complete
operating-system/BLAS environment.

Random streams are keyed by the study dimensions, replicate, and component
using SHA256, so changing the worker count does not change the assigned
streams. Floating-point results can differ slightly across library versions
and BLAS implementations. PDF files also contain generation metadata, so
byte-for-byte PDF equality is not the numerical reproduction criterion.
The saved raw draws and summaries preserve the exact results used in the
paper.

## Run on a Slurm cluster

Create or activate a Python environment accessible from the compute nodes,
then submit five separate jobs:

```bash
bash scripts/submit_all.sh --partition YOUR_PARTITION --cpus 6
```

Omit `--partition` to use the site's default. Additional options include
`--account`, `--mem`, `--time`, `--python`, and `--output-root`. For example,
on Unity with access to the `stat` partition:

```bash
bash scripts/submit_all.sh --partition stat --python "$VIRTUAL_ENV/bin/python"
```

The default allocation is six CPUs, 16 GB, and four hours **per study**.
The default output root is a fresh `runs/slurm-<timestamp>-<pid>/` directory.
Job IDs, stdout/stderr, exit status, and package versions are saved in its
`slurm/` subdirectory. Failed submissions stop the submission loop; any jobs
already accepted remain recorded in `slurm/submissions.tsv`.

Original Unity elapsed times were approximately 54 seconds (conditional),
5 minutes 46 seconds (accuracy), 3 minutes 38 seconds (threshold),
2 minutes 20 seconds (dense), and 4 minutes 25 seconds (isotropic), each with
six CPUs. These are historical measurements, not runtime guarantees for
another machine. Original job records are included in `provenance/`.

## Repository contents

```text
reproduce.py                    full-run and saved-summary rendering commands
readme.md                       setup and execution instructions
requirements.txt                runtime dependencies
environment-paper.txt           recorded direct-package versions
code/utils.py                   numerical routines and keyed random streams
code/main_studies/              five studies, shared model, table/plot code
scripts/                        portable Slurm submission scripts
docs/studies.md                 scientific protocol and output mapping
results/<study>/                reference raw draws, summaries, figures, metadata
results/conditional/frozen/     all 120 frozen compressed-data states
provenance/                     source snapshots, run logs, hashes, packaging record
runs/                           new run/render outputs (ignored by Git)
```

The `metadata.json` files and original submission records describe the
historical runs; their paths refer to the original computation environment.
`provenance/package.json` maps the bundled artifacts to those runs and records
the manuscript snapshot used for the output inventory.
`provenance/SHA256SUMS` records the bundled source and reference-result hashes.
On Linux, those files can be checked from the repository root with
`sha256sum --check provenance/SHA256SUMS`; on macOS, use
`shasum -a 256 --check provenance/SHA256SUMS`.

The published runner and Slurm scripts use only this repository's files and
the selected Python environment. They contain no author-specific compute
paths. Original cluster scripts and paths are retained only within the
historical source archives and run metadata.

During packaging, `python reproduce.py render` regenerated all five PDF/PNG
figures and all five table fragments. All five PNGs and all five table
fragments matched the archived paper versions byte for byte. No new Monte
Carlo simulations were run during packaging; the complete original runs and
their numerical source are included. The comparison hashes and rendering
environment are recorded in `provenance/package.json`.
