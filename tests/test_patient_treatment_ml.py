import os
import unittest
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestPatientTreatmentML(unittest.TestCase):
    def test_data_files_exist_and_non_empty(self):
        data_dir = os.path.join(repo_root, "data")
        self.assertTrue(os.path.isdir(data_dir), "data directory should exist")
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.assertGreater(len(csv_files), 0, "There should be at least one patient dataset CSV")

        for f in csv_files:
            file_path = os.path.join(data_dir, f)
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0, f"{f} should contain data rows")

    def test_agile_documentation_matrix(self):
        docs_dir = os.path.join(repo_root, "docs")
        self.assertTrue(os.path.isdir(docs_dir))
        self.assertTrue(os.path.exists(os.path.join(docs_dir, "ARCHITECTURE.md")))
        self.assertTrue(os.path.exists(os.path.join(docs_dir, "CURRICULUM_MATRIX.md")))


if __name__ == "__main__":
    unittest.main()
