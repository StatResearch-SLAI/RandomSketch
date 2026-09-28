# Scientific protocol

This repository reproduces the five simulation studies in Section 6 and
Appendix A of the manuscript. Every reported setting uses 3,000 repetitions.
The conditional study additionally uses 20 independently frozen fits per
setting; its repetitions are draws conditional on each fit.

## Model, estimators, and sketches

Data follow the linear model

\[
y=X\beta^*+\varepsilon,\qquad
X_i\sim N(0,\Sigma),\qquad \operatorname{Var}(\varepsilon_i)=1.
\]

Features, errors, and sketches are generated independently. Unless stated
otherwise, errors are Gaussian and
\(\Sigma_{jk}=0.5^{|j-k|}\). The first ten coefficients are equal, the
remaining coefficients are zero, and the common value is chosen so that
\(\beta^{*\top}\Sigma\beta^*=4\). The isotropic study instead uses
\(\Sigma=I_d\) and \(\|\beta^*\|^2=4\).

Each fitted predictor has the exact response-prediction risk

\[
R=1+(\widehat\beta-\beta^*)^\top\Sigma
       (\widehat\beta-\beta^*).
\]

The simulations evaluate this risk directly, without a finite test set.
They generate Gaussian feature and error coordinates, rather than drawing
risks or residuals from the scalar distributional laws in the proofs.

Write \(u=y-X\widehat\beta\), \(A=S^\top S\), and let
\(P_S\) project onto the row space of \(S\). The complementary estimate is
\(\widehat R_{\perp,S}=\|(I-P_S)u\|^2/(n-m)\). For \(d<m\), the
corrected estimate is
\(\widehat R_{\mathrm{obs}}=\|u+(d/b_A)Au\|^2/n\), where
\(b_A=\operatorname{tr}(A)-\operatorname{tr}\{AX(X^\top AX)^{-1}X^\top A\}\).
The population-sensitivity version replaces \(d/b_A\) with
\(t_\Sigma=\operatorname{tr}\{\Sigma(X^\top AX)^{-1}\}\).
Conventional GCV is
\(\{\|u\|^2/n\}/\{1-\min(d,m)/n\}^2\).

The sketch constructions are:

- **Row:** sample \(m\) rows uniformly without replacement and multiply
  the selection matrix by \(\sqrt{n/m}\).
- **Haar:** \(S=\sqrt{n/m}\,Q\), where \(Q\) is Haar distributed with
  orthonormal rows, \(QQ^\top=I_m\).
- **Weighted Haar:** \(S=c\operatorname{diag}(w_j)Q\), with alternating
  weights 1 and 2 and \(c^2=n/\sum_jw_j^2\).
- **Dense Gaussian:** independent \(S_{ij}\sim N(0,1/m)\), without
  trace normalization.

## 1. Conditional complementary risk estimation

Script: [`conditional.py`](../code/main_studies/conditional.py).

- Dimensions: \(n=256\), \(m=128\), \(d\in\{64,192,320\}\).
- Sketches: row and dense Gaussian; six settings.
- Seed: **2026092601**.
- Sampling: 20 frozen fits per setting, followed by 3,000 complementary
  data draws per fit; 120 frozen fits and 360,000 draws in total.

For each frozen fit, hold \((S,QX,Q\varepsilon)\) fixed, with
\(Q=(SS^\top)^{-1/2}S\), and fit the raw compressed equations. Generate
fresh Gaussian feature and error coordinates in the orthogonal complement.
Use least squares when \(d<m\) and the minimum-norm solution when \(d>m\).
Only for the interpolating fits does the full residual equal its
complementary component.

Metrics are the conditional mean of \(\widehat R_{\perp,S}/R\), its
sample variance divided by \(2/(n-m)\), and coverage of the exact nominal
95% chi-square risk interval. Compute metrics within each frozen fit, then
average over the 20 fits. Monte Carlo intervals use the standard error
across these independent frozen-fit clusters and the \(t_{19}\) quantile.
The frozen matrices are retained, with one full-data reconstruction refit
per frozen fit as a numerical diagnostic.

## 2. Accuracy of corrected risk estimation

Script: [`accuracy.py`](../code/main_studies/accuracy.py).

- \(n\in\{128,256,512\}\).
- \((d/n,m/n)\in\{(0.25,0.50),(0.50,0.75)\}\).
- Sketches: row and weighted Haar.
- Errors: Gaussian and independent Rademacher variables taking values
  \(-1\) and \(1\) with equal probabilities.
- Seed: **2026092601**; 24 settings and 72,000 fits.

Compare corrected estimation, population-sensitivity corrected estimation,
and GCV against the risk of their common fitted predictor. Complementary
estimation is summarized only for Gaussian errors. The primary metric is
MAE; the archive also reports bias, MSE, RMSE, \(\sqrt n\) times MAE, and
within-fit paired differences in absolute error.

Standard errors for means use the sample standard deviation divided by
\(\sqrt{3000}\). RMSE standard errors use the delta method applied to
mean squared error. Intervals use 1.96 standard errors. Tables report
standard errors in parentheses; bold paired differences have intervals
excluding zero before rounding. The main table uses \(n=512\); the four
appendix tables cover all 24 settings.

## 3. Tight-frame MSE threshold

Script: [`threshold.py`](../code/main_studies/threshold.py).

- \(n\in\{128,256,512\}\), \(d/n\in\{0.25,0.50\}\).
- Row sketches and Gaussian errors.
- \(m=\operatorname{round}\{n[\rho_*(d/n)+\delta]\}\), where
  \(\rho_*(\gamma)=3\gamma/(1+2\gamma)\) and
  \(\delta\in\{-0.08,-0.04,0,0.04,0.08\}\).
- Seed: **2026092601**; 30 settings, 90,000 fits from 18,000 datasets.

The empirical MSE ratio is corrected-estimate MSE divided by
complementary-estimate MSE, with both errors measured against the same
fit's exact risk. The 95% intervals use a log-scale delta method retaining
the pairing of squared errors. The figure also shows the exact MSE ratio,
\(\rho_*(\gamma)\), and the numerical root
\(\rho_n^{\mathrm{exact}}\) of the continuous finite-sample MSE equality.
This root differs from the first-order approximation
\(\rho_*(\gamma)-5/(12n)\). All settings satisfy \(d+3<m<n\).
The experiment assesses ordering away from equality; it does not resolve
the \(O(n^{-1})\) displacement between the thresholds.

## 4. Dense Gaussian benchmarks

Script: [`dense_benchmarks.py`](../code/main_studies/dense_benchmarks.py).

- \(n\in\{128,256,512\}\).
- \((d/n,m/n)\in\{(0.25,0.50),(0.50,0.75),(0.50,1.00)\}\).
- Raw dense Gaussian sketches, AR(1) features, Gaussian errors.
- Seed: **2026092701**; nine settings, 27,000 fits from 18,000 datasets.

Compare mean risk and the corrected and GCV estimates with the exact
finite-sample \(\mathbb ER\), the common risk limit, and the GCV limit.
The exact finite-sample mean curve applies to \(R\); it is not an
unbiasedness assertion for the corrected estimates. Also report the mean
absolute discrepancy \(|t_\Sigma-d/b_A|\), with intervals of 1.96 Monte
Carlo standard errors. Its decrease illustrates calibration without
establishing a convergence rate.

At \(m=n\), the square Gaussian sketch gives a random weighted
least-squares fit. The complementary estimate is undefined because there
is no complementary subspace; its missing raw value is intentional.
Keeping the stated raw sketch scale matters for the sensitivity quantities.

## 5. Isotropic minimum-norm benchmarks

Script: [`isotropic_benchmarks.py`](../code/main_studies/isotropic_benchmarks.py).

- \(n\in\{128,256,512\}\).
- \((d/n,m/n)\in\{(0.75,0.50),(1.25,0.50)\}\).
- Identity covariance, Gaussian errors, \(\|\beta^*\|^2=4\).
- Sketches: row, Haar, and raw dense Gaussian.
- Seed: **2026092701**; 18 settings, 54,000 fits from 18,000 datasets.

Fit the minimum-norm solution of the raw compressed equations. Compare
mean risk and mean complementary estimate with

\[
\mathbb ER=1+4(1-m/d)+\frac{m}{d-m-1},\qquad
\mathcal R_{\min}(\gamma,\rho)
=4(1-\rho/\gamma)+\frac{\gamma}{\gamma-\rho}.
\]

The concentration panels show empirical RMSE of \(R\) and of its
complementary estimate around this deterministic limit. These quantities
are distinct from estimation error relative to the realized risk of a
particular fit, which is recorded separately. Mean intervals use 1.96
standard errors; concentration RMSE intervals use delta-method standard
errors. The common benchmarks across sketch families concern equality in
distribution, rather than equality of fitted risks on paired datasets.

## Pairing, uncertainty, and numerical policy

- SHA256-keyed streams make draws independent of worker scheduling.
  Feature, error, and sketch streams are separated.
- Accuracy-study sketches share data, while its two error families share
  features and sketches. Threshold sketch sizes share each dataset and a
  nested row ordering.
- The six central Gaussian row-sketch settings in accuracy and threshold
  reuse the same **18,000 fits**. They are not independent replications.
- Dense settings with matching \((n,d)\) share data across sketch sizes.
  Isotropic sketch families share each dataset and use independent sketches.
  The supporting studies use fresh streams relative to the main studies.
- All reported intervals are pointwise; there is no simultaneous-coverage
  claim. Conditional uncertainty uses frozen-fit clusters; other studies
  use independent repetitions within each setting.
- Fits use QR without ridge regularization or singular-value truncation.
  Numerical failures remain in the records; no draw is replaced or trimmed.
  An incomplete run exits unsuccessfully and is not a completed paper
  reproduction. Use a fresh output directory for reruns.

## Manuscript output map

All study scripts are under `code/main_studies/`. Result paths below are
relative to the chosen results directory.

| Output | Study or formatter | Result artifact |
|---|---|---|
| Figure 1 | `conditional.py` | `conditional/figures/complementary_calibration.pdf` |
| Figure 2 | `accuracy.py` | `accuracy/figures/accuracy.pdf` |
| Table 2 | `accuracy.py` | `accuracy/accuracy_table.tex` |
| Figure 3 | `threshold.py` | `threshold/figures/threshold.pdf` |
| Table 3: bias | `make_appendix_tables.py` | `accuracy/appendix/bias_table.tex` |
| Table 4: RMSE | `make_appendix_tables.py` | `accuracy/appendix/rmse_table.tex` |
| Table 5: scaled MAE | `make_appendix_tables.py` | `accuracy/appendix/scaled_mae_table.tex` |
| Table 6: paired MAE differences | `make_appendix_tables.py` | `accuracy/appendix/paired_mae_table.tex` |
| Figure 4 | `dense_benchmarks.py` | `dense_benchmarks/figures/dense_benchmarks.pdf` |
| Figure 5 | `isotropic_benchmarks.py` | `isotropic_benchmarks/figures/isotropic_benchmarks.pdf` |

Table 1 summarizes theoretical regimes and has no simulation generator.
The appendix accuracy tables reuse the accuracy summaries and require
the paper's full 3,000 repetitions; they introduce no additional draws.
