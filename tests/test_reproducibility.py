import json
import tempfile
import unittest
from pathlib import Path

from src.reproducibility import make_metadata, seed_everything, trial_seed, write_json


class ReproducibilityTests(unittest.TestCase):
    def test_trial_seed_scheme(self):
        self.assertEqual([trial_seed(42, i) for i in range(3)], [42, 43, 44])

    def test_seed_repeats_numpy_stream(self):
        import numpy as np

        seed_everything(123)
        first = np.random.random(5)
        seed_everything(123)
        second = np.random.random(5)
        np.testing.assert_array_equal(first, second)

    def test_json_metadata_is_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"metadata": make_metadata({"seed": 42}), "values": [1, 2, 3]}
            write_json(path, payload)
            self.assertEqual(json.loads(path.read_text())["metadata"]["config"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
