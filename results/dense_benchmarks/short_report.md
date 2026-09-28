# Dense Gaussian risk and calibration benchmarks

Seed: 2026092701; repetitions per setting: 3,000. All 27,000 fits succeeded in nine settings.

Gaussian linear-model data use AR(1) covariance with correlation 0.5, noise variance one, and the common sparse signal with beta' Sigma beta = 4. Raw sketch entries are independent N(0,1/m), without trace normalization. Sample sizes are 128, 256, and 512; (d/n,m/n) are (0.25,0.50), (0.50,0.75), and (0.50,1.00).

The target is each fitted model's exact response-prediction risk. The fit uses full-rank QR without regularization or singular-value truncation. All methods share a fit. Matched (n,d,rep) settings share features and errors; sketch streams are independent. Repetitions within each setting are independent. No draws were replaced or trimmed.

The square raw Gaussian sketch at m=n is a random weighted least-squares fit. It is not the identity-sketch control. No complementary risk estimate is defined at that boundary; its raw-file entry is intentionally missing.

The figure contrasts empirical means with the exact finite-sample mean risk, the proportional risk limit, and the GCV limit. Its lower panels show mean absolute calibration discrepancy |t_Sigma-d/b_A|. All error bars are pointwise 95% Monte Carlo intervals (mean +/- 1.96 standard errors), without a simultaneous guarantee. The lower-panel absolute discrepancy is distinct from the signed difference of means.

| n | gamma | rho | Mean risk (MCSE) | Exact mean risk | Risk limit | Mean absolute calibration discrepancy (MCSE) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.25 | 0.50 | 2.3760 (0.0096) | 2.3800 | 2.3333 | 0.04734 (0.00069) |
| 256 | 0.25 | 0.50 | 2.3593 (0.0068) | 2.3563 | 2.3333 | 0.02352 (0.00034) |
| 512 | 0.25 | 0.50 | 2.3478 (0.0044) | 2.3447 | 2.3333 | 0.01205 (0.00016) |
| 128 | 0.50 | 0.75 | 4.1175 (0.0193) | 4.1132 | 4.0000 | 0.23197 (0.00335) |
| 256 | 0.50 | 0.75 | 4.0709 (0.0132) | 4.0556 | 4.0000 | 0.11016 (0.00154) |
| 512 | 0.50 | 0.75 | 4.0331 (0.0090) | 4.0276 | 4.0000 | 0.05371 (0.00075) |
| 128 | 0.50 | 1.00 | 3.0592 (0.0109) | 3.0479 | 3.0000 | 0.10312 (0.00146) |
| 256 | 0.50 | 1.00 | 3.0217 (0.0075) | 3.0237 | 3.0000 | 0.05003 (0.00070) |
| 512 | 0.50 | 1.00 | 3.0057 (0.0052) | 3.0118 | 3.0000 | 0.02568 (0.00035) |

## Saved outputs

- `raw.csv.gz`: every fit, draw IDs, risk estimates, exact model risk, calibration quantities, and estimation errors.
- `summary.csv`: per-setting means, Monte Carlo standard errors and intervals, MAEs, and exact/proportional benchmarks.
- `failures.csv`: retained failures (empty for a successful study).
- `figures/dense_benchmarks.pdf` and `.png`: six-panel benchmark figure.
- `metadata.json`: settings, counts, seed scheme, source hashes, and computing provenance.
