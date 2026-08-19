# data.py ingests and processes the GDSC dataset, providing functions to download, load, and validate the data.

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import pandas as pd


DOWNLOADS_URL = "https://gdsc-combinations.depmap.sanger.ac.uk/downloads"
TISSUE_DOWNLOADS = {
    "breast": ("/downloads/breast/anchor_combo",),
    "colon": ("/downloads/colon/anchor_combo",),
    "pancreas": ("/downloads/pancreas/anchor_combo",),
}
REQUIRED_COLUMNS = {
    "Cell Line name",
    "SDIM",
    "Tissue",
    "Cancer Type",
    "Anchor Name",
    "Anchor Target",
    "Anchor Pathway",
    "Anchor Conc",
    "Library Name",
    "library Target",
    "Library Pathway",
    "Maxc",
    "Library IC50",
    "Combo IC50",
    "Library Emax",
    "Bliss Emax",
    "Combo Emax",
    "Delta Xmid",
    "Delta Emax",
    "Synergy?",
}


class _DownloadLinkParser(HTMLParser):
    """Collect download links from the site's downloads catalog."""

    def __init__(self):
        super().__init__()
        self.links = []
        self._link_text = []
        self._href = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_text = []

    def handle_data(self, data):
        if self._href is not None:
            self._link_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((" ".join(self._link_text).strip(), self._href))
            self._href = None
            self._link_text = []


def download_gdsc(tissue_of_origin="Lung", data_dir="data/raw"):
    """Download every GDSC² dataset listed for a tissue of origin.

    The site's catalog can contain more than one dataset for a tissue. The
    returned paths are all matching files, saved beneath ``data_dir``.
    """
    parser = _DownloadLinkParser()
    with urlopen(DOWNLOADS_URL) as response:
        parser.feed(response.read().decode("utf-8"))

    requested_tissue = tissue_of_origin.casefold()
    matches = [
        (label, urljoin(DOWNLOADS_URL, href))
        for label, href in parser.links
        if requested_tissue in label.casefold()
        and href is not None
        and href.startswith("/downloads/")
    ]
    if not matches:
        matches = [
            (tissue_of_origin, urljoin(DOWNLOADS_URL, href))
            for href in TISSUE_DOWNLOADS.get(requested_tissue, ())
        ]
    if not matches:
        raise ValueError(
            f"No GDSC² download datasets found for tissue {tissue_of_origin!r}"
        )

    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []
    for label, url in matches:
        path_parts = [part for part in urlparse(url).path.split("/") if part]
        filename = "_".join(path_parts[-2:]) + ".csv.gz"
        output_path = output_dir / filename
        with urlopen(url) as response, output_path.open("wb") as output_file:
            output_file.write(response.read())
        downloaded_paths.append(output_path)
    return downloaded_paths


def load_gdsc(data_dir="data/raw", tissue_of_origin=None):
    """Load downloaded GDSC² datasets into one DataFrame.

    All compressed CSV files in ``data_dir`` are loaded in sorted filename
    order. If ``tissue_of_origin`` is provided, rows are filtered by ``Tissue``.
    """
    input_files = sorted(Path(data_dir).glob("*.csv.gz"))
    if not input_files:
        raise FileNotFoundError(f"No GDSC CSV files found in {data_dir!r}")

    data = pd.concat(
        (pd.read_csv(input_file, compression="gzip") for input_file in input_files),
        ignore_index=True,
    )
    data.columns = data.columns.str.strip()
    if tissue_of_origin is not None:
        if "Tissue" not in data.columns:
            raise ValueError("GDSC data does not contain a 'Tissue' column")
        data = data[data["Tissue"].str.casefold() == tissue_of_origin.casefold()]
        if data.empty:
            raise ValueError(
                f"No GDSC rows found for tissue {tissue_of_origin!r}"
            )
    return data


def validate_gdsc(data_dir="data/raw", tissue_of_origin=None, required_columns=None):
    """Validate local GDSC² files and return ``True`` when they are valid.

    By default, validation is tolerant of schema changes. When filtering by
    ``tissue_of_origin``, the ``Tissue`` column is required. Callers can pass
    ``required_columns`` for analysis-specific schema validation.
    """
    data = load_gdsc(data_dir)
    required_columns = set(required_columns or ())
    if tissue_of_origin is not None:
        required_columns.add("Tissue")
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"GDSC data is missing required columns: {missing}")

    if tissue_of_origin is not None:
        tissues = data["Tissue"].dropna().astype(str).str.casefold()
        if not (tissues == tissue_of_origin.casefold()).all():
            raise ValueError(
                f"GDSC data contains rows outside tissue {tissue_of_origin!r}"
            )
    return True
