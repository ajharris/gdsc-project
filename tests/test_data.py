import io
import gzip
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, "src")

from gdsc import data


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DownloadGdscTests(unittest.TestCase):
    def test_downloads_all_datasets_for_tissue(self):
        catalog = (
            b'<a href="/downloads/breast/anchor_combo">Breast anchor data</a>'
            b'<a href="/downloads/breast/extra">Breast extra data</a>'
            b'<a href="/downloads/colon/anchor_combo">Colon anchor data</a>'
        )
        responses = {
            data.DOWNLOADS_URL: catalog,
            "https://gdsc-combinations.depmap.sanger.ac.uk/downloads/breast/anchor_combo": b"first",
            "https://gdsc-combinations.depmap.sanger.ac.uk/downloads/breast/extra": b"second",
        }

        def fake_urlopen(url):
            return _Response(responses[url])

        with TemporaryDirectory() as directory, patch.object(
            data, "urlopen", side_effect=fake_urlopen
        ):
            paths = data.download_gdsc("bReAsT", directory)
            self.assertEqual(
                [path.name for path in paths],
                ["breast_anchor_combo.csv.gz", "breast_extra.csv.gz"],
            )
            self.assertEqual(
                (Path(directory) / "breast_anchor_combo.csv.gz").read_bytes(), b"first"
            )

    @patch.object(
        data,
        "urlopen",
        return_value=_Response(b'<a href="/downloads/colon/anchor_combo">Colon</a>'),
    )
    def test_raises_when_tissue_is_not_in_catalog(self, mock_urlopen):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Lung"):
                data.download_gdsc("Lung", directory)

        mock_urlopen.assert_called_once_with(data.DOWNLOADS_URL)


class LoadGdscTests(unittest.TestCase):
    def test_loads_and_filters_multiple_compressed_csv_files(self):
        with TemporaryDirectory() as directory:
            for filename, content in {
                "one.csv.gz": "Tissue,Score\nPancreas,1\n",
                "two.csv.gz": "Tissue,Score\nBreast,2\n",
            }.items():
                with gzip.open(Path(directory) / filename, "wt") as output_file:
                    output_file.write(content)

            loaded = data.load_gdsc(directory, tissue_of_origin="pAnCrEaS")

        self.assertEqual(loaded.to_dict("records"), [{"Tissue": "Pancreas", "Score": 1}])

    def test_raises_when_no_local_datasets_exist(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No GDSC CSV files"):
                data.load_gdsc(directory)

    def test_raises_when_tissue_column_is_missing(self):
        with TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "data.csv.gz", "wt") as output_file:
                output_file.write("Score\n1\n")

            with self.assertRaisesRegex(ValueError, "Tissue"):
                data.load_gdsc(directory, tissue_of_origin="pancreas")

    def test_raises_when_requested_tissue_has_no_rows(self):
        with TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "data.csv.gz", "wt") as output_file:
                output_file.write("Tissue,Score\nBreast,1\n")

            with self.assertRaisesRegex(ValueError, "Lung"):
                data.load_gdsc(directory, tissue_of_origin="Lung")


class ValidateGdscTests(unittest.TestCase):
    def _write_dataset(self, directory, rows):
        columns = sorted(data.REQUIRED_COLUMNS)
        content = ",".join(columns) + "\n"
        for row in rows:
            content += ",".join(row.get(column, "") for column in columns) + "\n"
        with gzip.open(Path(directory) / "data.csv.gz", "wt") as output_file:
            output_file.write(content)

    def test_returns_true_for_valid_dataset(self):
        with TemporaryDirectory() as directory:
            self._write_dataset(directory, [{"Tissue": "Pancreas"}])

            self.assertTrue(data.validate_gdsc(directory, "pancreas"))

    def test_raises_when_validation_directory_has_no_datasets(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No GDSC CSV files"):
                data.validate_gdsc(directory, tissue_of_origin="pancreas")

    def test_raises_when_required_columns_are_missing(self):
        with TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "data.csv.gz", "wt") as output_file:
                output_file.write("Tissue\nPancreas\n")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                data.validate_gdsc(directory, required_columns=data.REQUIRED_COLUMNS)

    def test_allows_additional_or_changed_columns_by_default(self):
        with TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "data.csv.gz", "wt") as output_file:
                output_file.write("Tissue,New Measurement\nPancreas,1.5\n")

            self.assertTrue(data.validate_gdsc(directory, tissue_of_origin="pancreas"))

    def test_raises_when_data_contains_another_tissue(self):
        with TemporaryDirectory() as directory:
            self._write_dataset(
                directory,
                [{"Tissue": "Pancreas"}, {"Tissue": "Breast"}],
            )

            with self.assertRaisesRegex(ValueError, "outside tissue"):
                data.validate_gdsc(directory, "pancreas")


class PrepareGdscTests(unittest.TestCase):
    @patch.object(data, "load_gdsc", return_value="loaded data")
    @patch.object(data, "validate_gdsc")
    @patch.object(data, "download_gdsc")
    def test_downloads_validates_and_loads(
        self, mock_download, mock_validate, mock_load
    ):
        result = data.prepare_gdsc(
            "pancreas",
            data_dir="tmp/raw",
            required_columns={"Tissue"},
        )

        self.assertEqual(result, "loaded data")
        mock_download.assert_called_once_with("pancreas", data_dir="tmp/raw")
        mock_validate.assert_called_once_with(
            "tmp/raw",
            tissue_of_origin="pancreas",
            required_columns={"Tissue"},
        )
        mock_load.assert_called_once_with("tmp/raw", tissue_of_origin="pancreas")


if __name__ == "__main__":
    unittest.main()
