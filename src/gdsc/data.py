"""Acquire and load the single-agent GDSC release 8.4 data."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

import pandas as pd

load_dotenv()  # Load COSMIC_LINK from .env if present

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

COSMIC_EXPRESSION_ARCHIVE = (
    "CellLinesProject_CompleteGeneExpression_Tsv_v104_GRCh38.tar"
)
COSMIC_EXPRESSION_FILE = (
    "CellLinesProject_CompleteGeneExpression_v104_GRCh38.tsv.gz"
)

RESPONSE_REQUIRED_COLUMNS = {
    "DATASET",
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "DRUG_NAME",
    "AUC",
    "LN_IC50",
}

EXPRESSION_REQUIRED_COLUMNS = {
    "COSMIC_SAMPLE_ID",
    "SAMPLE_NAME",
    "COSMIC_GENE_ID",
    "GENE_SYMBOL",
    "REGULATION",
    "Z_SCORE",
    "COSMIC_STUDY_ID",
}

MANIFEST_NAME = "gdsc_release_manifest.json"


def _source_url(filename: str) -> str:
    return f"{SOURCE_ROOT}/{filename}"


def _download_file(url: str, output_path: Path) -> None:
    """Download a URL to a local file."""
    with urlopen(url) as response, output_path.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)

def _download_cosmic_file(url: str, output_path: Path) -> None:
    """Download a COSMIC signed URL using curl."""
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--output",
            str(output_path),
            url,
        ],
        check=True,
    )

def _cosmic_link() -> str:
    """Return the COSMIC expression download URL from the environment."""
    link = os.environ.get("COSMIC_LINK")
    if not link:
        raise RuntimeError(
            "COSMIC_LINK is not set. Add the current signed COSMIC expression "
            "download URL to the environment."
        )
    return link


def _extract_cosmic_expression(archive_path: Path, output_dir: Path) -> Path:
    """Extract the COSMIC expression TSV from the downloaded archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    expression_path = output_dir / COSMIC_EXPRESSION_FILE
    if expression_path.exists():
        return expression_path

    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()

        matching_members = [
            member
            for member in members
            if Path(member.name).name == COSMIC_EXPRESSION_FILE
        ]

        if len(matching_members) != 1:
            raise ValueError(
                "Expected exactly one COSMIC expression file in archive; "
                f"found {len(matching_members)}"
            )

        member = matching_members[0]

        if not member.isfile():
            raise ValueError(
                f"COSMIC expression archive member is not a regular file: "
                f"{member.name}"
            )

        archive.extract(member, path=output_dir)

    extracted_path = output_dir / member.name

    if extracted_path != expression_path:
        expression_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.replace(expression_path)

    return expression_path


def download_gdsc(data_dir="../data/raw") -> dict[str, Path]:
    """Download the GDSC 8.4 response/metadata files and COSMIC expression.

    The GDSC release is pinned to release 8.4. The COSMIC expression source
    is supplied through the COSMIC_LINK environment variable because COSMIC
    provides a user-specific, time-limited signed URL.

    Existing files are reused, so repeated calls do not redownload data.
    """
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: dict[str, Path] = {}

    # GDSC release files.
    for key, filename in SOURCE_FILES.items():
        output_path = output_dir / filename

        if not output_path.exists():
            _download_file(_source_url(filename), output_path)

        downloaded_files[key] = output_path

    # COSMIC expression archive.
    archive_path = output_dir / COSMIC_EXPRESSION_ARCHIVE

    if not archive_path.exists():
        _download_cosmic_file(_cosmic_link(), archive_path)

    downloaded_files["cosmic_expression_archive"] = archive_path

    # Extract the expression TSV.
    expression_path = _extract_cosmic_expression(
        archive_path,
        output_dir,
    )

    downloaded_files["cosmic_expression"] = expression_path

    manifest = {
        "release": RELEASE,
        "release_date": RELEASE_DATE,
        "source_root": SOURCE_ROOT,
        "files": SOURCE_FILES,
        "cosmic_expression": {
            "product": "Cell Lines Project Complete Gene Expression",
            "version": "v104",
            "genome_build": "GRCh38",
            "archive": COSMIC_EXPRESSION_ARCHIVE,
            "expression_file": COSMIC_EXPRESSION_FILE,
        },
    }

    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    return downloaded_files


def _response_files(data_dir: str | Path) -> list[Path]:
    return sorted(
        Path(data_dir).glob("GDSC[12]_fitted_dose_response_*.csv")
    )


def _load_cell_line_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(
        metadata_path,
        sheet_name="Cell line details",
    )

    metadata.columns = metadata.columns.str.strip()

    metadata = metadata.rename(
        columns={
            "GDSC\nTissue descriptor 1": "TISSUE_OF_ORIGIN",
            "GDSC\nTissue\ndescriptor 2": "TISSUE_DESCRIPTOR_2",
            "Cancer Type\n(matching TCGA label)": "CANCER_TYPE",
        }
    )

    metadata["COSMIC_ID"] = pd.to_numeric(
        metadata["COSMIC identifier"],
        errors="coerce",
    ).astype("Int64")

    return metadata


def _load_cosmic_expression(expression_path: Path) -> pd.DataFrame:
    """Load COSMIC v104 expression data in long format."""
    expression = pd.read_csv(
        expression_path,
        sep="\t",
        compression="gzip",
    )

    expression.columns = expression.columns.str.strip()

    missing = EXPRESSION_REQUIRED_COLUMNS - set(expression.columns)

    if missing:
        raise ValueError(
            "COSMIC expression data is missing columns: "
            f"{sorted(missing)}"
        )

    return expression


def _prepare_cosmic_expression(
    expression_path: Path,
) -> pd.DataFrame:
    """Convert COSMIC expression data from long to cell-line-wide format.

    Rows in the COSMIC source represent:

        COSMIC sample × gene

    The returned DataFrame contains one row per SAMPLE_NAME and one column
    per GENE_SYMBOL. Z_SCORE is used as the expression value.
    """
    expression = _load_cosmic_expression(expression_path)

    expression = expression[
        [
            "COSMIC_SAMPLE_ID",
            "SAMPLE_NAME",
            "GENE_SYMBOL",
            "Z_SCORE",
        ]
    ].copy()

    expression["SAMPLE_NAME"] = (
        expression["SAMPLE_NAME"]
        .astype("string")
        .str.strip()
    )

    expression["GENE_SYMBOL"] = (
        expression["GENE_SYMBOL"]
        .astype("string")
        .str.strip()
    )

    expression["Z_SCORE"] = pd.to_numeric(
        expression["Z_SCORE"],
        errors="coerce",
    )

    duplicate_pairs = expression.duplicated(
        subset=["COSMIC_SAMPLE_ID", "GENE_SYMBOL"],
        keep=False,
    )

    if duplicate_pairs.any():
        duplicates = expression.loc[
            duplicate_pairs,
            ["COSMIC_SAMPLE_ID", "GENE_SYMBOL"],
        ].drop_duplicates()

        raise ValueError(
            "COSMIC expression contains duplicate sample/gene pairs. "
            f"Found {len(duplicates):,} duplicated pairs."
        )

    expression_wide = expression.pivot(
        index=["COSMIC_SAMPLE_ID", "SAMPLE_NAME"],
        columns="GENE_SYMBOL",
        values="Z_SCORE",
    ).reset_index()

    expression_wide.columns.name = None

    return expression_wide


def _join_cosmic_expression(
    response_data: pd.DataFrame,
    metadata: pd.DataFrame,
    expression_path: Path,
) -> pd.DataFrame:
    """Join COSMIC expression to GDSC using the curated sample names.

    GDSC uses the numeric COSMIC_ID, whereas the COSMIC expression product
    uses COSMIC_SAMPLE_ID (COSS...). The GDSC cell-line workbook and COSMIC
    expression product share SAMPLE_NAME. We therefore establish the
    mapping through SAMPLE_NAME and retain the COSMIC_SAMPLE_ID as provenance.
    """
    expression = _prepare_cosmic_expression(expression_path)

    metadata_sample_names = (
        metadata["Sample Name"]
        .astype("string")
        .str.strip()
    )

    # Ignore non-cell-line summary rows such as "TOTAL:".
    valid_metadata = metadata[
        metadata_sample_names.notna()
        & metadata_sample_names.ne("")
        & metadata_sample_names.str.upper().ne("TOTAL:")
    ].copy()

    valid_metadata["Sample Name"] = (
        valid_metadata["Sample Name"]
        .astype("string")
        .str.strip()
    )

    # Confirm that each COSMIC sample name maps to one COSMIC sample.
    sample_counts = (
        expression.groupby("SAMPLE_NAME")["COSMIC_SAMPLE_ID"]
        .nunique()
    )

    ambiguous_names = sample_counts[sample_counts > 1]

    if not ambiguous_names.empty:
        raise ValueError(
            "COSMIC SAMPLE_NAME maps to multiple COSMIC_SAMPLE_ID values: "
            f"{ambiguous_names.index.tolist()[:10]}"
        )

    # Establish GDSC Sample Name -> COSMIC_SAMPLE_ID.
    sample_mapping = expression[
        ["COSMIC_SAMPLE_ID", "SAMPLE_NAME"]
    ].drop_duplicates("SAMPLE_NAME")

    metadata_with_expression = valid_metadata.merge(
        sample_mapping,
        left_on="Sample Name",
        right_on="SAMPLE_NAME",
        how="left",
        validate="one_to_one",
    )

    # Add expression features using the COSMIC sample identifier.
    result = response_data.merge(
        metadata_with_expression[
            [
                "COSMIC_ID",
                "COSMIC_SAMPLE_ID",
            ]
        ],
        on="COSMIC_ID",
        how="left",
        validate="many_to_one",
    )

    expression_features = expression.drop(
        columns=["SAMPLE_NAME"],
    )

    result = result.merge(
        expression_features,
        on="COSMIC_SAMPLE_ID",
        how="left",
        validate="many_to_one",
    )

    return result


def load_gdsc(
    data_dir="../data/raw",
    include_metadata=True,
    include_expression=False,
) -> pd.DataFrame:
    """Load GDSC responses with optional metadata and expression data.

    GDSC1 and GDSC2 responses are concatenated.

    When ``include_metadata`` is true, cell-line tissue and cancer metadata
    from the release workbook are left-joined by COSMIC_ID.

    When ``include_expression`` is true, COSMIC v104 gene-expression
    Z-scores are joined using the GDSC/COSMIC Sample Name mapping.
    """
    response_paths = _response_files(data_dir)

    if not response_paths:
        raise FileNotFoundError(
            f"No GDSC fitted response CSV files found in {data_dir!r}"
        )

    response_data = pd.concat(
        (
            pd.read_csv(path)
            for path in response_paths
        ),
        ignore_index=True,
    )

    response_data.columns = response_data.columns.str.strip()

    if not include_metadata and not include_expression:
        return response_data

    metadata_path = Path(data_dir) / SOURCE_FILES["cell_lines"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"GDSC cell-line metadata not found: {metadata_path}"
        )

    metadata = _load_cell_line_metadata(metadata_path)

    data = response_data.merge(
        metadata,
        on="COSMIC_ID",
        how="left",
        suffixes=("", "_METADATA"),
        validate="many_to_one",
    )

    if include_expression:
        expression_path = (
            Path(data_dir) / COSMIC_EXPRESSION_FILE
        )

        if not expression_path.exists():
            raise FileNotFoundError(
                "COSMIC expression data not found. "
                "Run download_gdsc() first."
            )

        data = _join_cosmic_expression(
            response_data=data,
            metadata=metadata,
            expression_path=expression_path,
        )

    return data


def filter_by_tissue(
    data: pd.DataFrame,
    tissue_of_origin: str,
) -> pd.DataFrame:
    """Return rows matching a tissue without changing downloaded data."""
    if "TISSUE_OF_ORIGIN" not in data.columns:
        raise ValueError(
            "Data does not contain TISSUE_OF_ORIGIN metadata"
        )

    tissue = (
        data["TISSUE_OF_ORIGIN"]
        .astype("string")
        .str.casefold()
    )

    filtered = data[
        tissue == tissue_of_origin.casefold()
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No GDSC rows found for tissue {tissue_of_origin!r}"
        )

    return filtered


def validate_gdsc(data_dir="../data/raw") -> bool:
    """Validate source files and structural fields needed for analysis."""
    data_path = Path(data_dir)

    paths = {
        key: data_path / filename
        for key, filename in SOURCE_FILES.items()
    }

    missing_files = [
        path.name
        for path in paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing GDSC release files in ../data/raw: "
            f"{missing_files}"
        )

    for key in ("gdsc1_response", "gdsc2_response"):
        columns = set(
            pd.read_csv(
                paths[key],
                nrows=0,
            ).columns.str.strip()
        )

        missing = RESPONSE_REQUIRED_COLUMNS - columns

        if missing:
            raise ValueError(
                f"GDSC response data is missing columns: "
                f"{sorted(missing)}"
            )

    metadata = _load_cell_line_metadata(
        paths["cell_lines"]
    )

    required_metadata = {
        "COSMIC_ID",
        "TISSUE_OF_ORIGIN",
        "CANCER_TYPE",
        "Sample Name",
    }

    missing = required_metadata - set(metadata.columns)

    if missing:
        raise ValueError(
            f"GDSC metadata is missing columns: {sorted(missing)}"
        )

    expression_path = data_path / COSMIC_EXPRESSION_FILE

    if not expression_path.exists():
        raise FileNotFoundError(
            f"COSMIC expression data not found: {expression_path}"
        )

    # Validate expression structure without loading all expression values.
    expression_columns = set(
        pd.read_csv(
            expression_path,
            sep="\t",
            compression="gzip",
            nrows=0,
        ).columns.str.strip()
    )

    missing = EXPRESSION_REQUIRED_COLUMNS - expression_columns

    if missing:
        raise ValueError(
            "COSMIC expression data is missing columns: "
            f"{sorted(missing)}"
        )

    return True


def prepare_gdsc(
    data_dir="../data/raw",
    include_expression=False,
) -> pd.DataFrame:
    """Download, validate, and load the complete single-agent release."""
    download_gdsc(data_dir)
    validate_gdsc(data_dir)

    return load_gdsc(
        data_dir,
        include_metadata=True,
        include_expression=include_expression,
    )