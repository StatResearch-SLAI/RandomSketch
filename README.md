# Simulation code for *Prediction Risk After Observation Compression*

This code reproduces the simulations in Section 5 and Appendix A, including
Figures 1–6 and Tables 1–3. All data are simulated; no external dataset or
LaTeX installation is needed. Saved simulation results are not included in
the standalone code repository.

Run the commands below from the directory containing `reproduce.py`: the root of the standalone
code repository.

## Install

Use Python 3.9 or newer. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The files `environment-paper.txt` and `environment-risk-curves.txt` record
the direct package versions used for the original runs. They can replace
`requirements.txt` when installing in a compatible Python environment.
Small floating-point differences can occur across package versions and
BLAS implementations.

## Run the simulations

Run all four studies:

```bash
python reproduce.py run --jobs 6
```

The studies run one after another, using up to six workers per study.
Reduce `--jobs` to suit your machine. Full runs involve many fitted models
and can take substantial time.

| Study (`--study`) | What it checks | Manuscript outputs |
|---|---|---|
| `conditional` | Conditional calibration of the complementary risk estimate | Figure 1 |
| `risk_curves` | Prediction risk and risk estimates across three sketch families | Figures 2 and 4; Tables 1–3 |
| `threshold` | Estimation MSE for row sketches | Figure 3 |
| `threshold_shapes` | Estimation MSE for weighted Haar and dense Gaussian sketches | Figures 5–6 |

To run only one study and choose a new output location:

```bash
python reproduce.py run --study risk_curves --jobs 6 --output-root runs/risk-rerun
```

The runner uses the manuscript's fixed parameter grids and random seeds,
with 1,000 repetitions per setting. The conditional study uses 100 frozen
fits per setting and 1,000 complementary-data draws per fit. See Section 5
and Appendix A for the simulation design.

## Find the results

By default, each study writes to `runs/paper/<study>/`. A custom
`--output-root` replaces `runs/paper/`. Outputs include raw records,
summaries, PDF/PNG figures, run metadata, and, for `risk_curves`, LaTeX
table fragments.

The runner validates completed results before accepting them. Failed draws
are retained without replacement; failed or incomplete runs are rejected.
Existing nonempty study directories are not overwritten. For a new run,
choose a fresh `--output-root`.

## Regenerate figures and tables

Running a study already creates its figures and tables. To rebuild them
later from a completed run, without generating new data:

```bash
python reproduce.py render
```

This reads `runs/paper/` and writes to `runs/rendered/`. For a single study
in a different location:

```bash
python reproduce.py render --study risk_curves \
  --input-root runs/risk-rerun --output-root runs/risk-rendered
```

Inputs must contain complete, validated results. Input and output roots
must be separate directories, with neither inside the other. The output
study directories must be empty or absent.

## Resume an interrupted run

Only `risk_curves` and `threshold_shapes` support resuming from saved
checkpoints. For runs started with the default output location:

```bash
python code/main_studies/risk_curves.py resume \
  --output runs/paper/risk_curves --jobs 6

python code/main_studies/threshold_shapes.py --resume \
  --output runs/paper/threshold_shapes --jobs 6
```

For a custom run location, change `--output` to the interrupted study's
directory. Resuming requires the same simulation settings, generating
source files, and package versions. The worker count may change.
For interrupted `conditional` or `threshold` runs, restart with a fresh
output location.

## Validate conditional results independently

The main runner validates all four studies automatically. To rerun the
independent conditional-study checks separately:

```bash
python validate_conditional.py --report runs/paper/conditional --paper
```

This checks the saved models, raw records, checkpoints, and summaries,
and replays part of the data generation.

## Reproduce all simulation outputs from this folder

After installing the dependencies, run the following command from the
directory containing `reproduce.py`:

```bash
python reproduce.py run --jobs 6
```

This generates the simulated data, Figures 1–6, and Tables 1–3 using only
the files in this folder. Results are saved under `runs/paper/` in four
study directories: `conditional/`, `risk_curves/`, `threshold/`, and
`threshold_shapes/`. Keep these directories, including their raw records,
checkpoints, and metadata, to validate or render the results later.
Figures are saved as PDF and PNG files in each study's `figures/`
directory. The three LaTeX table fragments are saved directly in
`runs/paper/risk_curves/`.

The initial run already creates the figures and tables. To rebuild them
from your completed results without generating new data, optionally run:

```bash
python reproduce.py render
```

The rebuilt figures and tables are saved under `runs/rendered/`, organized
by study. For another simulation run, choose a fresh `--output-root` as
shown above.
