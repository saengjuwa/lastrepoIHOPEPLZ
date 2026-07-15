from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import windows_app


class TrainingCsvStatusTests(unittest.TestCase):
    def test_detects_partial_gldv2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv.part"
            path.write_text("id,url,landmark_id\na,http://example.com/a.jpg,1\n")
            status, _ = windows_app.training_csv_status(str(path))
        self.assertEqual(status, "gldv2_metadata_partial")

    def test_detects_original_gldv2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text("id,url,landmark_id\na,http://example.com/a.jpg,1\n")
            status, _ = windows_app.training_csv_status(str(path))
        self.assertEqual(status, "gldv2_metadata")

    def test_detects_ready_manifest_and_country_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            without_country = root / "without_country.csv"
            with_country = root / "with_country.csv"
            without_country.write_text("path,label\na.jpg,1\n")
            with_country.write_text("path,label,country_code\na.jpg,1,FR\n")

            status_without, _ = windows_app.training_csv_status(str(without_country))
            status_with, _ = windows_app.training_csv_status(str(with_country))

        self.assertEqual(status_without, "ready_needs_korean_labels")
        self.assertEqual(status_with, "ready_with_country")


if __name__ == "__main__":
    unittest.main()
