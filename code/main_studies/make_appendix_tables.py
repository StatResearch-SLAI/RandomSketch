#!/usr/bin/env python3
"""Create appendix tables from the completed Section 6.2 summary CSVs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METHODS = ("corrected", "population_sensitivity_corrected", "gcv", "complementary")
LABELS = (r"$\hR_{\mathrm{obs}}$", r"$\hR_{\mathrm{sens}}$",
          r"$\hR_{\GCV}$", r"$\widehat R_{\perp,\bS}$")
RATIOS = ((0.25, 0.50), (0.50, 0.75))


def number(value, digits):
    result = f"{value:.{digits}f}"
    return result.removeprefix("-") if float(result) == 0 else result


def table(frame, metric=None):
    paired = metric is None
    methods = METHODS[1:] if paired else METHODS
    labels = LABELS[1:] if paired else LABELS
    columns = 3 + len(methods)
    lines = [r"\begin{tabular}{@{}rll" + "r" * len(methods) + r"@{}}",
             r"\toprule", " $n$ & Sketch & Errors & " + " & ".join(labels) + r" \\",
             r"\midrule"]
    for panel, (gamma, rho) in enumerate(RATIOS):
        if panel:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(columns) + r"}{c}{$(d/n,m/n)=("
                     + f"{gamma:.2f},{rho:.2f}" + r")$} \\")
        lines.append(r"\midrule")
        for n in (128, 256, 512):
            if n != 128:
                lines.append(r"\addlinespace[3pt]")
            for family in ("row", "weighted_haar"):
                for noise in ("gaussian", "rademacher"):
                    group = frame[(frame.n == n) & (frame.gamma == gamma)
                                  & (frame.rho == rho) & (frame.sketch == family)
                                  & (frame.noise == noise)]
                    cells = []
                    for method in methods:
                        key = "second_method" if paired else "method"
                        row = group[group[key] == method]
                        if row.empty:
                            if not (noise == "rademacher" and method == "complementary"):
                                raise ValueError(f"Missing output: {n}, {gamma}, {family}, {noise}, {method}")
                            cells.append("---")
                            continue
                        if len(row) != 1 or int(row.iloc[0]["count"]) != 3000:
                            raise ValueError("Appendix requires one complete 3000-draw summary per cell")
                        row = row.iloc[0]
                        if paired:
                            value = number(row.paired_mae_difference, 4)
                            if row.ci_high < 0 or row.ci_low > 0:
                                value = r"\mathbf{" + value + "}"
                            cells.append("$" + value + r"$ (" + number(row.difference_mcse, 4) + ")")
                        else:
                            cells.append("$" + number(row[metric], 3) + "$ ("
                                         + number(row[metric + "_mcse"], 3) + ")")
                    first = family == "row" and noise == "gaussian"
                    name = "Row" if family == "row" else "Weighted Haar"
                    error = "Gaussian" if noise == "gaussian" else "Rademacher"
                    lines.append((str(n) if first else "") + f" & {name} & {error} & "
                                 + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="Completed accuracy-study directory")
    args = parser.parse_args()
    root = args.report.resolve()
    metrics = pd.read_csv(root / "summary.csv")
    comparisons = pd.read_csv(root / "paired_comparisons.csv")
    comparisons = comparisons[comparisons.first_method == "corrected"]
    output = root / "appendix"
    output.mkdir(exist_ok=True)
    for metric in ("bias", "rmse", "scaled_mae"):
        (output / f"{metric}_table.tex").write_text(table(metrics, metric))
    (output / "paired_mae_table.tex").write_text(table(comparisons))
    (output / "provenance.json").write_text(json.dumps({
        "inputs": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("summary.csv", "paired_comparisons.csv")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": 24, "repetitions_per_setting": 3000,
        "method_setting_rows_per_metric": len(metrics),
        "paired_corrected_comparisons": len(comparisons),
        "operation": "Format existing summaries; no simulation rerun",
        "paired_bold": "Pointwise 95% paired interval excludes zero before rounding",
    }, indent=2) + "\n")
    print(f"Wrote four appendix tables to {output}")


if __name__ == "__main__":
    main()
