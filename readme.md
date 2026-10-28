# RandomSketch: prediction risk after observation compression

The current manuscript reproduction bundle is in [Code_CPAL](Code_CPAL/README.md).
It covers the four studies in Section 5 and Appendix A of
*Prediction Risk After Observation Compression*.

| Study | Current manuscript outputs |
|---|---|
| `conditional` | Figure 1: conditional calibration |
| `risk_curves` | Figures 2 and 4; Tables 1–3: mean risk, paired bias, and RMSE |
| `threshold` | Figure 3: row-sampling MSE comparison |
| `threshold_shapes` | Figures 5 and 6: weighted Haar and dense Gaussian MSE comparisons |

## Reproduce the current manuscript

Use Python 3.9 or newer in a virtual environment:

```bash
git clone https://github.com/StatResearch-SLAI/RandomSketch.git
cd RandomSketch/Code_CPAL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py run --jobs 6
python reproduce.py render
```

The runner uses the manuscript's fixed grids and seeds. Both MSE studies and
the mean-risk study use 1,000 repetitions per setting. The conditional study
uses 100 frozen fits per setting and 1,000 complementary draws per fit.
`run` saves new results under `Code_CPAL/runs/paper/`; `render` validates
completed runs and writes figures and tables under `Code_CPAL/runs/rendered/`.
Existing nonempty output directories are rejected.

The [bundle README](Code_CPAL/README.md) gives the full protocols, per-study
commands, validation, checkpoints, resume requirements, and recorded environments.
No external dataset or LaTeX installation is required for these simulations.

The current completed-run archives referenced as `reports/...` in that README
belong to the manuscript checkout and are not included in this standalone
code distribution. Use `run` above before `render` in a fresh clone.
The reference files under this repository's root `results/` are historical
and use earlier grids; they are separate from the current bundle's results.

## Historical archive

The root `code/`, `reproduce.py`, `scripts/`, `results/`, and original
`provenance/` records retain the September 2026 studies. Their commands and
output mapping are documented in the
[historical reproduction guide](docs/historical-reproduction-20260928.md).
Use `Code_CPAL/` for the current manuscript.

The immutable original package is available at
[commit a7bdabc](https://github.com/StatResearch-SLAI/RandomSketch/tree/a7bdabc81d82eaa9b191cb2be7ef628943e120a1).
The later September package remains available at
[commit 45d9e04](https://github.com/StatResearch-SLAI/RandomSketch/tree/45d9e04a2a898fb69226c93ec063858215ee77c5).

`provenance/cpal-source-manifest.json` records hashes of the current bundle.
`provenance/SHA256SUMS` covers the retained files and the new code package.
From the repository root, check it on Linux with
`sha256sum --check provenance/SHA256SUMS`, or on macOS with
`shasum -a 256 --check provenance/SHA256SUMS`.

