"""Portable paths, installed reference data, and measured CLI accuracy."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from caims.__main__ import main
from caims.reference import REFERENCE_RESULTS


class ReleaseTests(unittest.TestCase):
    def test_default_data_root_is_relative_to_working_directory_and_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.pop("CAIMS_DATA_ROOT", None)
            # Include the checkout only when running tests without installation.
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            command = [sys.executable, "-c", "from caims.paths import DATA_ROOT; print(DATA_ROOT)"]
            default = subprocess.check_output(command, cwd=temporary, env=environment, text=True).strip()
            self.assertEqual(Path(default), Path(temporary) / "data")
            environment["CAIMS_DATA_ROOT"] = "custom-data"
            overridden = subprocess.check_output(command, cwd=temporary, env=environment, text=True).strip()
            self.assertEqual(Path(overridden), Path(temporary) / "custom-data")

    def test_reference_data_is_in_package(self):
        self.assertTrue(REFERENCE_RESULTS.is_file())
        self.assertIn("T_select_DASS", REFERENCE_RESULTS.read_text(encoding="utf-8"))

    def run_evaluated_search(self, qrels):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, output = root / "paper.json", root / "matches.jsonl"
            vectors = (np.zeros((3, 24), dtype=np.uint8), np.zeros((2, 24), dtype=np.float32),
                       np.array([b"a", b"b", b"c"]), ["test-query", "unlabeled-query"])
            captured = io.StringIO()
            with mock.patch("caims.quantization.load_vectors", return_value=vectors), \
                 mock.patch("caims.metrics.load_qrels", return_value=qrels), \
                 contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()):
                main(["paper-config", "--dataset", "scifact", "--output", str(config)])
                main(["last", "--config", str(config), "--output", str(output),
                      "--device", "cpu", "--evaluate"])
            return json.loads(next(line.removeprefix("Evaluation: ")
                                   for line in captured.getvalue().splitlines()
                                   if line.startswith("Evaluation: ")))

    def test_evaluation_uses_measured_accuracy_and_labels_only_for_denominator(self):
        summary = self.run_evaluated_search({"test-query": {"not-retrieved": 1}})
        self.assertEqual(summary["hit_rate_percent"], 0)
        self.assertEqual(summary["evaluated_queries"], 1)
        self.assertEqual(summary["encoded_queries"], 2)
        self.assertEqual(summary["mean_result_count_all_queries"], 3)

    def test_evaluation_rejects_an_empty_labeled_population(self):
        with self.assertRaisesRegex(ValueError, "No queries have evaluation labels"):
            self.run_evaluated_search({})
