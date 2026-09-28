#!/usr/bin/env python3
"""Render the three manuscript figures from completed simulation summaries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import matplotlib
import pandas as pd

import accuracy
import conditional
import threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.report.resolve()
    conditional.make_figure(pd.read_csv(root / "conditional/settings.csv"),
                            root / "conditional")
    accuracy.make_figure(pd.read_csv(root / "accuracy/summary.csv"), root / "accuracy")
    threshold.make_figure(pd.read_csv(root / "threshold/summary.csv"), root / "threshold")
    presentation = root / "presentation"
    source = presentation / "source"
    source.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, source / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (presentation / "metadata.json").write_text(json.dumps({
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Re-render saved summary CSVs; no simulation rerun",
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "source_sha256": hashes,
        "original_figures": "presentation/unity-original/",
        "simulation_source": "source/code/",
    }, indent=2) + "\n")
    print(f"Rendered the three manuscript figures from {root}")


if __name__ == "__main__":
    main()
