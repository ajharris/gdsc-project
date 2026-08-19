import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, "src")

from gdsc import data


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _write_response(path, columns=None):
    columns = columns or sorted(data.RESPONSE_REQUIRED_COLUMNS)
    row = {column: "value" for column in columns}
    row.update(
        {
            column: value
            for column, value in {
                "COSMIC_ID": 123,
                "AUC": 0.5,
                "LN_IC50": 1.2,
            }.items()
            if column in row
        }
    )
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_metadata(path, columns=None):
    columns = columns or [
        "Sample Name",
        "COSMIC identifier",
        "GDSC\nTissue descriptor 1",
        "GDSC\nTissue\ndescriptor 2",
        "Cancer Type\n(matching TCGA label)",
    ]
    row = {column: "value" for column in columns}
    row.update(
        {
            "Sample Name": "MODEL-1",
            "COSMIC identifier": 123,
            "GDSC\nTissue descriptor 1": "pancreas",
            "GDSC\nTissue descriptor 2": "pancreas",
            "Cancer Type\n(matching TCGA label)": "Pancreatic carcinoma",
        }
    )
    pd.DataFrame([row]).to_excel(path, sheet_name="Cell line details", index=False)


class DownloadGdscTests(unittest.TestCase):
    def test_downloads_pinned_release_files_and_writes_manifest(self):
        responses = {
            data._source_url(filename): filename.encode()
            for filename in data.SOURCE_FILES.values()
        }

        def fake_urlopen(url):
            return _Response(responses[url])

        with TemporaryDirectory() as directory, patch.object(
            data, "urlopen", side_effect=fake_urlopen
        ):
            paths = data.download_gdsc(directory)
            self.assertEqual(set(paths), set(data.SOURCE_FILES))
            manifest = json.loads(
                (Path(directory) / data.MANIFEST_NAME).read_text()
            )
            self.assertEqual(manifest["release"], data.RELEASE)
            self.assertTrue(all(path.exists() for path in paths.values()))

    def test_reuses_existing_files_without_redownloading(self):
        with TemporaryDirectory() as directory:
            for filename in data.SOURCE_FILES.values():
                (Path(directory) / filename).write_bytes(b"cached")

            with patch.object(data, "urlopen") as mock_urlopen:
                paths = data.download_gdsc(directory)

        mock_urlopen.assert_not_called()
        self.assertEqual(set(paths), set(data.SOURCE_FILES))


class LoadGdscTests(unittest.TestCase):
    def test_loads_both_response_files_and_merges_metadata(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            _write_response(directory / data.SOURCE_FILES["gdsc1_response"])
            _write_response(directory / data.SOURCE_FILES["gdsc2_response"])
            _write_metadata(directory / data.SOURCE_FILES["cell_lines"])

            loaded = data.load_gdsc(directory)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded["TISSUE_OF_ORIGIN"].tolist(), ["pancreas", "pancreas"])
        self.assertEqual(loaded["CANCER_TYPE"].iloc[0], "Pancreatic carcinoma")

    def test_can_load_responses_without_metadata(self):
        with TemporaryDirectory() as directory:
            _write_response(Path(directory) / data.SOURCE_FILES["gdsc1_response"])

            loaded = data.load_gdsc(directory, include_metadata=False)

        self.assertEqual(len(loaded), 1)
        self.assertNotIn("TISSUE_OF_ORIGIN", loaded)

    def test_raises_when_response_files_are_missing(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "fitted response"):
                data.load_gdsc(directory)

    def test_raises_when_metadata_is_missing(self):
        with TemporaryDirectory() as directory:
            _write_response(Path(directory) / data.SOURCE_FILES["gdsc1_response"])

            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                data.load_gdsc(directory)


class TissueFilterTests(unittest.TestCase):
    def test_filters_without_mutating_the_input(self):
        frame = pd.DataFrame(
            {
                "TISSUE_OF_ORIGIN": ["pancreas", "breast"],
                "AUC": [0.1, 0.2],
            }
        )

        filtered = data.filter_by_tissue(frame, "PanCrEaS")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(frame), 2)

    def test_raises_when_tissue_metadata_is_missing(self):
        with self.assertRaisesRegex(ValueError, "TISSUE_OF_ORIGIN"):
            data.filter_by_tissue(pd.DataFrame({"AUC": [0.1]}), "pancreas")

    def test_raises_when_tissue_has_no_rows(self):
        frame = pd.DataFrame({"TISSUE_OF_ORIGIN": ["breast"]})
        with self.assertRaisesRegex(ValueError, "pancreas"):
            data.filter_by_tissue(frame, "pancreas")


class ValidateGdscTests(unittest.TestCase):
    def test_reports_missing_release_files_without_network_access(self):
        with TemporaryDirectory() as directory, patch.object(data, "urlopen") as mock_urlopen:
            with self.assertRaisesRegex(FileNotFoundError, "Missing GDSC release files"):
                data.validate_gdsc(directory)

        mock_urlopen.assert_not_called()

    def test_validates_response_and_metadata_schema(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            _write_response(directory / data.SOURCE_FILES["gdsc1_response"])
            _write_response(directory / data.SOURCE_FILES["gdsc2_response"])
            _write_metadata(directory / data.SOURCE_FILES["cell_lines"])

            self.assertTrue(data.validate_gdsc(directory))

    def test_reports_missing_response_column(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            columns = sorted(data.RESPONSE_REQUIRED_COLUMNS - {"AUC"})
            _write_response(directory / data.SOURCE_FILES["gdsc1_response"], columns)
            _write_response(directory / data.SOURCE_FILES["gdsc2_response"])
            _write_metadata(directory / data.SOURCE_FILES["cell_lines"])

            with self.assertRaisesRegex(ValueError, "AUC"):
                data.validate_gdsc(directory)


class PrepareGdscTests(unittest.TestCase):
    @patch.object(data, "load_gdsc", return_value="loaded")
    @patch.object(data, "validate_gdsc")
    @patch.object(data, "download_gdsc")
    def test_downloads_validates_then_loads(
        self, mock_download, mock_validate, mock_load
    ):
        result = data.prepare_gdsc("tmp/raw")

        self.assertEqual(result, "loaded")
        mock_download.assert_called_once_with("tmp/raw")
        mock_validate.assert_called_once_with("tmp/raw")
        mock_load.assert_called_once_with("tmp/raw")


if __name__ == "__main__":
    unittest.main()
