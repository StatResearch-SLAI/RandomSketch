# Accuracy of corrected risk estimation

Seed: 2026092601. Independent repetitions per setting: 3,000.

The 24 settings cross n = 128, 256, 512; (d/n,m/n) = (0.25,0.50), (0.50,0.75); row and weighted Haar sketches; and Gaussian and Rademacher errors. The Gaussian design has AR(1) covariance with correlation 0.5. Noise variance is one. The coefficient vector is fixed and sparse, with beta' Sigma beta = 4. Weighted Haar sketches have alternating row weights 1 and 2 and trace(S' S) = n.

Every method targets the exact model risk 1 + (beta_hat-beta)' Sigma (beta_hat-beta) of its own fitted model. Methods and sketches share each data draw; Gaussian and Rademacher errors share the feature draw and sketch. Replicates are independent; changing the number of workers does not change their random streams. Where dimensions coincide, the Gaussian row-sketch draws are also shared with the threshold study. That overlap is a paired reuse of data, not an independent replication.

The corrected estimate uses d/b_A; the population-sensitivity corrected estimate uses the known population covariance. GCV uses the full residual and d degrees of freedom. Complementary risk estimates are summarized only under Gaussian errors. They remain in the raw file for both error distributions, but no exact Gaussian calibration claim is made for the Rademacher values.

Attempted fits: 72,000; numerical failures: 0. Failures are retained with their draw IDs and reason; no draw is replaced and no successful value is trimmed. Each metric lists attempted, available and unavailable counts.

## Paired comparisons

A negative paired MAE difference favors the first method. Intervals are the mean paired absolute-error difference plus or minus 1.96 Monte Carlo standard errors. They are pointwise intervals over independent replicates, without simultaneous coverage.

Against GCV, the corrected estimate has a paired 95% interval below zero in 24/24 settings, above zero in 0/24, and overlapping zero in 0/24.

Against population-sensitivity corrected estimate, the corrected estimate has a paired 95% interval below zero in 21/24 settings, above zero in 0/24, and overlapping zero in 3/24.

## Files

- `raw.csv.gz`: all attempted fits, exact model risks, estimates, errors, and draw IDs.
- `summary.csv`: bias, MAE, MSE, RMSE, Monte Carlo standard errors, and sqrt(n) MAE.
- `paired_comparisons.csv`: all within-fit paired MAE comparisons and 95% intervals.
- `accuracy_table.tex`: n = 512 MAE (Monte Carlo standard error) table.
- `figures/accuracy.pdf` and `.png`: MAE versus n; bars are pointwise 95% Monte Carlo intervals.
- `failures.csv`: all failures, including draw IDs (empty if there were none).
- `metadata.json`: settings, counts, random seed, and provenance.
