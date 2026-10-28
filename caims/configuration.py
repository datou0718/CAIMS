"""Read/write search configurations using the paper's parameter names."""

import json
from pathlib import Path

SCHEMA_VERSION = 2


def read_configuration(path: Path) -> dict:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    version = configuration.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported search configuration schema: {version}")
    for key in ("encoder", "dataset", "bits", "array_width", "k", "t_filter", "t_select"):
        if key not in configuration:
            raise ValueError(f"Missing search configuration field: {key}")
    return configuration


def write_configuration(path: Path, configuration: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
