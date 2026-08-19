"""Acquire and load the single-agent GDSC release 8.4 data."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


RELEASE = "8.4"
RELEASE_DATE = "2022-07-24"
SOURCE_ROOT = (
    "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/"
    f"release-{RELEASE}"
)
SOURCE_FILES = {
    "gdsc1_response": "GDSC1_fitted_dose_response_24Jul22.csv",
    "gdsc2_response": "GDSC2_fitted_dose_response_24Jul22.csv",
    "cell_lines": "Cell_Lines_Details.xlsx",
}
RESPONSE_REQUIRED_COLUMNS = {
    "DATASET",
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "DRUG_NAME",
    "AUC",
    "LN_IC50",
}
MANIFEST_NAME = "gdsc_release_manifest.json"


def _source_url(filename: str) -> str:
    return f"{SOURCE_ROOT}/{filename}"


def _download_file(url: str, output_path: Path) -> None:
    with urlopen(url) as response, output_path.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)


def download_gdsc(data_dir="data/raw") -> dict[str, Path]:
    """Download the pinned single-agent GDSC 8.4 source files.

    Tissue is deliberately not an argument: the release is downloaded in
    full and tissue metadata is retained for later analysis. Existing files
    are reused, so repeated calls do not redownload them.
    """
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = {}
    for key, filename in SOURCE_FILES.items():
        output_path = output_dir / filename
        if not output_path.exists():
            _download_file(_source_url(filename), output_path)
        downloaded_files[key] = output_path

    manifest = {
        "release": RELEASE,
        "release_date": RELEASE_DATE,
        "source_root": SOURCE_ROOT,
        "files": SOURCE_FILES,
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return downloaded_files


def _response_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("GDSC[12]_fitted_dose_response_*.csv"))


def _load_cell_line_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(metadata_path, sheet_name="Cell line details")
    metadata.columns = metadata.columns.str.strip()
    metadata = metadata.rename(
        columns={
            "GDSC\nTissue descriptor 1": "TISSUE_OF_ORIGIN",
            "GDSC\nTissue\ndescriptor 2": "TISSUE_DESCRIPTOR_2",
            "Cancer Type\n(matching TCGA label)": "CANCER_TYPE",
        }
    )
    metadata["COSMIC_ID"] = pd.to_numeric(
        metadata["COSMIC identifier"], errors="coerce"
    ).astype("Int64")
    return metadata


def load_gdsc(data_dir="data/raw", include_metadata=True) -> pd.DataFrame:
    """Load all cached single-agent response files into one DataFrame.

    GDSC1 and GDSC2 responses are concatenated. When ``include_metadata`` is
    true, cell-line tissue and cancer metadata from the release workbook are
    left-joined by ``COSMIC_ID``.
    """
    response_paths = _response_files(data_dir)
    if not response_paths:
        raise FileNotFoundError(
            f"No GDSC fitted response CSV files found in {data_dir!r}"
        )
    response_data = pd.concat(
        (pd.read_csv(path) for path in response_paths), ignore_index=True
    )
    response_data.columns = response_data.columns.str.strip()
    if not include_metadata:
        return response_data

    metadata_path = Path(data_dir) / SOURCE_FILES["cell_lines"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"GDSC cell-line metadata not found: {metadata_path}")
    metadata = _load_cell_line_metadata(metadata_path)
    return response_data.merge(
        metadata, on="COSMIC_ID", how="left", suffixes=("", "_METADATA")
    )


def filter_by_tissue(data: pd.DataFrame, tissue_of_origin: str) -> pd.DataFrame:
    """Return rows matching a tissue without changing downloaded data."""
    if "TISSUE_OF_ORIGIN" not in data.columns:
        raise ValueError("Data does not contain TISSUE_OF_ORIGIN metadata")
    tissue = data["TISSUE_OF_ORIGIN"].astype("string").str.casefold()
    filtered = data[tissue == tissue_of_origin.casefold()].copy()
    if filtered.empty:
        raise ValueError(f"No GDSC rows found for tissue {tissue_of_origin!r}")
    return filtered


def validate_gdsc(data_dir="data/raw") -> bool:
    """Validate source files and structural fields needed for analysis."""
    data_path = Path(data_dir)
    paths = {
        key: data_path / filename for key, filename in SOURCE_FILES.items()
    }
    missing_files = [path.name for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing GDSC release files in {data_dir!r}: {missing_files}"
        )
    for key in ("gdsc1_response", "gdsc2_response"):
        columns = set(pd.read_csv(paths[key], nrows=0).columns.str.strip())
        missing = RESPONSE_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"GDSC response data is missing columns: {sorted(missing)}"
            )

    metadata = _load_cell_line_metadata(paths["cell_lines"])
    required_metadata = {"COSMIC_ID", "TISSUE_OF_ORIGIN", "CANCER_TYPE"}
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"GDSC metadata is missing columns: {sorted(missing)}")
    return True


def prepare_gdsc(data_dir="data/raw") -> pd.DataFrame:
    """Download, validate, and load the complete single-agent release."""
    download_gdsc(data_dir)
    validate_gdsc(data_dir)
    return load_gdsc(data_dir)
