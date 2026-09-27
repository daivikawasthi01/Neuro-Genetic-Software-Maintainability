import tempfile
import unittest
from pathlib import Path

from src.data_collector import MiningCache, RepositoryHistoryIndex


class MiningTests(unittest.TestCase):
    def test_cache_rejects_different_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            config = {"schema_version": 2, "source_commit": "abc", "snapshot": "x"}
            cache = MiningCache(str(path), config)
            cache.put("module.py", {"loc": 10})
            cache.save()

            self.assertEqual(MiningCache(str(path), config).get("module.py"), {"loc": 10})
            changed = {**config, "source_commit": "def"}
            self.assertIsNone(MiningCache(str(path), changed).get("module.py"))

    def test_history_index_groups_records_by_path(self):
        index = RepositoryHistoryIndex("test_repos/flask")
        self.assertEqual(len(index.source_commit), 40)
        self.assertTrue(index.records_for("src/flask/app.py"))


if __name__ == "__main__":
    unittest.main()
