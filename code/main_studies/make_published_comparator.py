#!/usr/bin/env python3
"""Compare Bellec et al.'s full-data estimate on archived row-sampling fits.

For ordinary least squares on m selected rows, equation (13) in Section 3.1,
Remark 2 of https://pratikpatil.io/papers/cgcv.pdf specializes to
||u||^2 / (n - 2d + d^2/m). The archived ``raw`` field is ||u||^2/n.
This script adds no fits or random draws and leaves the input files unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


REPETITIONS = 3000
GROUP_KEYS = ["n", "d", "m", "gamma", "rho", "sketch", "noise"]
SAMPLE_SIZES = (128, 256, 512)
RATIOS = ((0.25, 0.50), (0.50, 0.75))
NOISES = ("gaussian", "rademacher")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize(raw):
    """Retain within-fit pairing and reject missing or failed published cells."""
    frame = raw.loc[raw.sketch == "row"].copy()
    expected = {(n, round(n * gamma), round(n * rho), gamma, rho, "row", noise)
                for n in SAMPLE_SIZES for gamma, rho in RATIOS for noise in NOISES}
    observed = set(frame[GROUP_KEYS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("The published comparison requires all 12 row-sampling settings")
    rows = []
    for key, group in frame.groupby(GROUP_KEYS, sort=True, dropna=False):
        base = dict(zip(GROUP_KEYS, key))
        if (len(group) != REPETITIONS or group.rep.nunique() != REPETITIONS
                or not group.status.eq("ok").all()):
            raise ValueError(f"Incomplete or failed row-sampling setting: {base}")
        values = group[["raw", "risk", "corrected"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite archived result: {base}")
        n, d, m = base["n"], base["d"], base["m"]
        denominator = n - 2 * d + d ** 2 / m
        if not (d < m < n and denominator > 0):
            raise ValueError(f"Setting outside the row-sampling comparison: {base}")
        estimate = n * group.raw.to_numpy(dtype=float) / denominator
        published_error = np.abs(estimate - group.risk.to_numpy(dtype=float))
        corrected_error = np.abs(group.corrected.to_numpy(dtype=float)
                                 - group.risk.to_numpy(dtype=float))
        published_mae, published_se = mean_mcse(published_error)
        corrected_mae, corrected_se = mean_mcse(corrected_error)
        difference, difference_se = mean_mcse(corrected_error - published_error)
        rows.append(dict(base, count=len(group),
                         corrected_mae=corrected_mae,
                         corrected_mae_mcse=corrected_se,
                         published_mae=published_mae,
                         published_mae_mcse=published_se,
                         paired_mae_difference=difference,
                         difference_mcse=difference_se,
                         ci_low=difference - 1.96 * difference_se,
                         ci_high=difference + 1.96 * difference_se))
    return pd.DataFrame(rows)


def number(value, digits):
    result = f"{value:.{digits}f}"
    return result.removeprefix("-") if float(result) == 0 else result


def table(summary):
    lines = [r"\begingroup",
             r"\begin{tabular}{@{}rlrrr@{}}", r"\toprule",
             r"$n$ & Errors & \shortstack{Corrected\\estimate} & "
             r"\shortstack{Bellec et al.\\full-data estimate} & "
             r"\shortstack{Paired MAE\\difference} \\", r"\midrule"]
    for panel, (gamma, rho) in enumerate(RATIOS):
        if panel:
            lines.append(r"\midrule")
        lines += [r"\multicolumn{5}{c}{$(d/n,m/n)=("
                  + f"{gamma:.2f},{rho:.2f}" + r")$} \\", r"\midrule"]
        for n in SAMPLE_SIZES:
            if n != SAMPLE_SIZES[0]:
                lines.append(r"\addlinespace[3pt]")
            for noise in NOISES:
                row = summary[(summary.n == n) & (summary.gamma == gamma)
                              & (summary.rho == rho) & (summary.noise == noise)].iloc[0]
                difference = number(row.paired_mae_difference, 4)
                if row.ci_high < 0 or row.ci_low > 0:
                    difference = r"\mathbf{" + difference + "}"
                cells = [number(row.corrected_mae, 3) + " ("
                         + number(row.corrected_mae_mcse, 3) + ")",
                         number(row.published_mae, 3) + " ("
                         + number(row.published_mae_mcse, 3) + ")",
                         "$" + difference + "$ (" + number(row.difference_mcse, 4) + ")"]
                error_name = "Gaussian" if noise == "gaussian" else "Rademacher"
                lines.append((str(n) if noise == NOISES[0] else "")
                             + f" & {error_name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="Completed accuracy-study directory containing raw.csv.gz")
    parser.add_argument("--output", type=Path,
                        help="Destination (default: REPORT/published_comparator)")
    args = parser.parse_args()
    source = args.report.resolve() / "raw.csv.gz"
    output = args.output.resolve() if args.output else source.parent / "published_comparator"
    frame = pd.read_csv(source, usecols=GROUP_KEYS + ["rep", "status", "raw", "risk", "corrected"])
    summary = summarize(frame)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "mae_table.tex").write_text(table(summary))
    (output / "provenance.json").write_text(json.dumps({
        "operation": "Evaluate published comparator on archived fits; no new simulation draws",
        "input_sha256": {"raw.csv.gz": digest(source)},
        "source_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "settings": len(summary), "repetitions_per_setting": REPETITIONS,
        "fits": int(summary["count"].sum()),
        "comparator": "Bellec et al. full-data estimate",
        "reference": "https://pratikpatil.io/papers/cgcv.pdf, Section 3.1, Remark 2, equation (13)",
        "formula": "n * raw / (n - 2*d + d*d/m), where raw = ||u||^2/n",
        "scope": "Ordinary least squares with literal row subsampling only",
        "difference": "Corrected absolute error minus published-comparator absolute error, paired within fit",
        "interval": "Pointwise normal interval: difference +/- 1.96 * sample SD / sqrt(3000)",
        "paired_bold": "Pointwise 95% paired interval excludes zero before rounding",
    }, indent=2) + "\n")
    print(f"Wrote {len(summary)} published-comparator summaries and a table to {output}")


if __name__ == "__main__":
    main()
