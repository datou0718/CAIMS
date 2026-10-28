"""Paper parameter replay using the current configuration format."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from caims.__main__ import main
from caims.configuration import read_configuration
from caims.paths import DATASETS
from caims.reference import paper_configuration


class ReferenceConfigurationTests(unittest.TestCase):
    def test_every_paper_dataset_loads_final_parameters_and_explicit_accuracy_units(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                configuration = paper_configuration(dataset)
                self.assertEqual(configuration['source'], 'paper_reference')
                self.assertTrue(configuration['t_filter'] < 0)
                self.assertTrue(0 < configuration['t_select'] <= configuration['k'])
                self.assertTrue(0 < configuration['reference']['reported_hit_rate_at_100'] <= 1)
        # The table's rounded accuracy and the recorded run retain separate precision.
        quora = paper_configuration('quora')['reference']
        self.assertAlmostEqual(quora['reported_hit_rate_at_100'], .9989)
        self.assertAlmostEqual(quora['table_iii_hit_rate_at_100'], .999)

    def test_paper_config_replays_through_last_without_recalibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration, result = root / 'paper.json', root / 'matches.jsonl'
            candidates = np.zeros((3, 24), dtype=np.uint8)
            vectors = (candidates, candidates[:1].astype(np.float32),
                       np.array([b'one', b'two', b'three']), ['query'])
            with mock.patch('caims.dass.select_parameters', side_effect=AssertionError('unexpected DASS')), \
                 mock.patch('caims.orca.adapt_t_select', side_effect=AssertionError('unexpected ORCA')), \
                 mock.patch('caims.quantization.load_vectors', return_value=vectors), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(['paper-config', '--dataset', 'scifact', '--output', str(configuration)])
                main(['last', '--config', str(configuration), '--output', str(result), '--device', 'cpu'])
            self.assertEqual(read_configuration(configuration)['t_select'], 80)
            self.assertEqual(json.loads(result.read_text()),
                             {'query_id': 'query', 'document_ids': ['one', 'two', 'three']})
