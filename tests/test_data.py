import io
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


if __name__ == "__main__":
    unittest.main()
