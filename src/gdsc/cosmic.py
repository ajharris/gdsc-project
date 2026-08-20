"""Acquire, process, cache, and load COSMIC cell-line gene expression data."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


COSMIC_EXPRESSION_ARCHIVE = (
    "CellLinesProject_CompleteGeneExpression_Tsv_v104_GRCh38.tar"
)
COSMIC_EXPRESSION_FILE = (
    "CellLinesProject_CompleteGeneExpression_v104_GRCh38.tsv.gz"
)
COSMIC_EXPRESSION_PARQUET = "cosmic_expression.parquet"

EXPRESSION_REQUIRED_COLUMNS = {
    "COSMIC_SAMPLE_ID",
    "SAMPLE_NAME",
    "COSMIC_GENE_ID",
    "GENE_SYMBOL",
    "REGULATION",
    "Z_SCORE",
    "COSMIC_STUDY_ID",
}


def _cosmic_link() -> str:
    """Return the current user-specific COSMIC download URL."""
    link = os.environ.get("COSMIC_LINK")

    if not link:
        raise RuntimeError(
            "COSMIC_LINK is not set. Add the current signed COSMIC "
            "expression download URL to the environment."
        )

    return link


def _download_file(url: str, output_path: Path) -> None:
    """Download a URL to a local file."""
    with urlopen(url) as response, output_path.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)


def _extract_expression_archive(
    archive_path: Path,
    output_dir: Path,
) -> Path:
    """Extract the COSMIC expression TSV from the downloaded archive."""
    expression_path = output_dir / COSMIC_EXPRESSION_FILE

    if expression_path.exists():
        return expression_path

    with tarfile.open(archive_path, mode="r") as archive:
        members = [
            member
            for member in archive.getmembers()
            if Path(member.name).name == COSMIC_EXPRESSION_FILE
        ]

        if len(members) != 1:
            raise ValueError(
                "Expected exactly one COSMIC expression file in archive; "
                f"found {len(members)}"
            )

        member = members[0]

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


def download_cosmic_expression(
    data_dir="../data/raw",
) -> dict[str, Path]:
    """Download and extract the COSMIC v104 expression dataset.

    The COSMIC download URL is supplied through COSMIC_LINK because it is
    user-specific and time-limited.
    """
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / COSMIC_EXPRESSION_ARCHIVE
    expression_path = output_dir / COSMIC_EXPRESSION_FILE

    if not archive_path.exists():
        _download_file(_cosmic_link(), archive_path)

    if not expression_path.exists():
        _extract_expression_archive(
            archive_path,
            output_dir,
        )

    return {
        "archive": archive_path,
        "expression": expression_path,
    }


def _load_expression_tsv(
    expression_path: Path,
) -> pd.DataFrame:
    """Load the raw COSMIC expression TSV."""
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


def _prepare_expression(
    expression_path: Path,
) -> pd.DataFrame:
    """Convert COSMIC expression from long to cell-line-wide format.

    COSMIC provides one record per sample/gene/study. The analytical
    expression matrix contains one row per COSMIC sample and one column
    per gene, with Z_SCORE as the expression value.

    COSU619 contains a small number of duplicated sample/gene records.
    These arise from duplicate observations within the source study.
    When duplicate observations have different Z-scores, their mean is
    used so that each sample/gene pair has a single reproducible value.
    """
    expression = _load_expression_tsv(expression_path)

    expression = expression[
        [
            "COSMIC_SAMPLE_ID",
            "SAMPLE_NAME",
            "GENE_SYMBOL",
            "Z_SCORE",
            "COSMIC_STUDY_ID",
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

    # Keep only the expression value needed for the ML feature matrix.
    # COSMIC_STUDY_ID is retained during duplicate resolution but is not
    # part of the final feature matrix.
    expression = (
        expression.groupby(
            [
                "COSMIC_SAMPLE_ID",
                "SAMPLE_NAME",
                "GENE_SYMBOL",
            ],
            as_index=False,
        )["Z_SCORE"]
        .mean()
    )

    expression_wide = expression.pivot(
        index=[
            "COSMIC_SAMPLE_ID",
            "SAMPLE_NAME",
        ],
        columns="GENE_SYMBOL",
        values="Z_SCORE",
    ).reset_index()

    expression_wide.columns.name = None

    return expression_wide


def load_or_build_expression(
    data_dir="../data/raw",
) -> pd.DataFrame:
    """Load cached COSMIC expression or build the Parquet cache.

    If the Parquet cache exists, the expensive raw TSV processing step is
    skipped.
    """
    data_path = Path(data_dir)
    parquet_path = data_path / COSMIC_EXPRESSION_PARQUET

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    expression_path = data_path / COSMIC_EXPRESSION_FILE

    if not expression_path.exists():
        download_cosmic_expression(data_path)

    expression = _prepare_expression(expression_path)

    expression.to_parquet(
        parquet_path,
        index=False,
    )

    return expression


def map_cosmic_samples(
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Map GDSC cell-line metadata to COSMIC sample identifiers.

    The mapping uses the shared Sample Name/SAMPLE_NAME field. The
    resulting table contains the GDSC COSMIC_ID and corresponding
    COSMIC_SAMPLE_ID.
    """
    required_metadata = {
        "COSMIC_ID",
        "Sample Name",
    }

    missing = required_metadata - set(metadata.columns)

    if missing:
        raise ValueError(
            "GDSC metadata is missing columns required for COSMIC mapping: "
            f"{sorted(missing)}"
        )

    metadata_mapping = metadata[
        [
            "COSMIC_ID",
            "Sample Name",
        ]
    ].copy()

    metadata_mapping["Sample Name"] = (
        metadata_mapping["Sample Name"]
        .astype("string")
        .str.strip()
    )

    metadata_mapping = metadata_mapping[
        metadata_mapping["Sample Name"].notna()
        & metadata_mapping["Sample Name"].ne("")
        & metadata_mapping["Sample Name"].str.upper().ne("TOTAL:")
    ]

    sample_mapping = expression[
        [
            "COSMIC_SAMPLE_ID",
            "SAMPLE_NAME",
        ]
    ].drop_duplicates()

    sample_counts = (
        sample_mapping.groupby("SAMPLE_NAME")["COSMIC_SAMPLE_ID"]
        .nunique()
    )

    ambiguous = sample_counts[sample_counts > 1]

    if not ambiguous.empty:
        raise ValueError(
            "COSMIC SAMPLE_NAME maps to multiple COSMIC_SAMPLE_ID values: "
            f"{ambiguous.index.tolist()[:10]}"
        )

    mapping = metadata_mapping.merge(
        sample_mapping,
        left_on="Sample Name",
        right_on="SAMPLE_NAME",
        how="left",
        validate="one_to_one",
    )

    return mapping[
        [
            "COSMIC_ID",
            "COSMIC_SAMPLE_ID",
        ]
    ].drop_duplicates("COSMIC_ID")


def join_expression(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Join COSMIC expression features onto GDSC response data."""
    mapping = map_cosmic_samples(
        metadata,
        expression,
    )

    result = data.merge(
        mapping,
        on="COSMIC_ID",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        expression,
        on="COSMIC_SAMPLE_ID",
        how="left",
        validate="many_to_one",
    )

    return result