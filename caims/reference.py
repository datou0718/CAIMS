"""Load recorded paper parameters without performing DASS or ORCA calibration."""

import csv
from importlib.resources import files
from pathlib import Path

from .configuration import SCHEMA_VERSION
REFERENCE_RESULTS = files("caims").joinpath("data", "reference_results.csv")


def paper_configuration(dataset: str, path: Path = REFERENCE_RESULTS) -> dict:
    with path.open(encoding="utf-8", newline="") as source:
        matches = [row for row in csv.DictReader(source) if row["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paper reference for {dataset}, found {len(matches)}")
    row = matches[0]
    return {
        "schema_version": SCHEMA_VERSION, "source": "paper_reference",
        "dataset": dataset, "encoder": row["encoder"],
        "bits": int(row["bits"]), "array_width": int(row["array_width"]),
        "k": int(row["k"]), "t_filter": float(row["T_filter"]),
        "t_select": int(row["T_select"]),
        "reference": {
            "file": str(path), "benchmark": row["benchmark"],
            "t_select_dass": int(row["T_select_DASS"]),
            "dass_calibration_c": float(row["dass_calibration_mean_result_count"]),
            "reported_hit_rate_at_100": float(row["hit_rate_at_100_percent"]) / 100,
            "reported_c": float(row["mean_result_count"]),
            "table_iii_hit_rate_at_100": float(row["table_iii_hit_rate_at_100_percent"]) / 100,
        },
    }
