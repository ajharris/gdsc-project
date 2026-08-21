"""Memory-safe COSMIC Cell Lines expression feature store.

Raw COSMIC files remain in ``data/raw``. A de-duplicated long-format Parquet
cache is built in ``data/processed`` and queried by sample and/or gene. This
module never attaches all expression features to all GDSC response rows.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Resolve configuration from the project, not the caller's working directory;
# notebooks execute from ``notebook/`` while command-line use often starts at root.
load_dotenv(PROJECT_ROOT / ".env.example")
load_dotenv(PROJECT_ROOT / ".env")
COSMIC_EXPRESSION_ARCHIVE = os.environ["COSMIC_EXPRESSION_ARCHIVE"]
COSMIC_EXPRESSION_FILE = os.environ["COSMIC_EXPRESSION_FILE"]
COSMIC_SAMPLE_FILE = os.environ["COSMIC_SAMPLE_FILE"]
COSMIC_EXPRESSION_PARQUET = os.environ["COSMIC_EXPRESSION_PARQUET"]
COSMIC_EXPRESSION_URL = os.environ["COSMIC_EXPRESSION_URL"]
EXPRESSION_COLUMNS = ["COSMIC_SAMPLE_ID", "SAMPLE_NAME", "GENE_SYMBOL", "Z_SCORE"]
EXPRESSION_REQUIRED_COLUMNS = set(EXPRESSION_COLUMNS) | {"COSMIC_GENE_ID", "REGULATION", "COSMIC_STUDY_ID"}


def _cosmic_request() -> Request:
    """Build an authenticated request without logging credentials."""
    link = os.environ.get("COSMIC_LINK", COSMIC_EXPRESSION_URL)
    authorization = os.environ.get("COSMIC_AUTHORIZATION")
    if link == COSMIC_EXPRESSION_URL and not authorization:
        raise RuntimeError("COSMIC_AUTHORIZATION is not set; add it to local .env")
    return Request(link, headers={"Authorization": f"Basic {authorization}"} if authorization else {})


def _download_file(request: Request | str, destination: Path) -> None:
    with urlopen(request) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _extract_member(archive_path: Path, output_dir: Path, filename: str) -> Path:
    destination = output_dir / filename
    if destination.exists():
        return destination
    with tarfile.open(archive_path) as archive:
        matches = [member for member in archive.getmembers() if Path(member.name).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {filename} in archive; found {len(matches)}")
        member = matches[0]
        if not member.isfile():
            raise ValueError(f"COSMIC archive member is not a regular file: {member.name}")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Cannot read COSMIC archive member: {member.name}")
        with source, destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
    return destination


def download_cosmic_expression(data_dir="data/raw") -> dict[str, Path]:
    """Download/reuse COSMIC v104 expression archive and extract its TSV."""
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / COSMIC_EXPRESSION_ARCHIVE
    if not archive.exists():
        _download_file(_cosmic_request(), archive)
    return {"archive": archive, "expression": _extract_member(archive, raw_dir, COSMIC_EXPRESSION_FILE)}


def _normalise_expression(frame: pd.DataFrame) -> pd.DataFrame:
    missing = EXPRESSION_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"COSMIC expression data is missing columns: {sorted(missing)}")
    frame = frame.loc[:, EXPRESSION_COLUMNS].copy()
    frame["SAMPLE_NAME"] = frame["SAMPLE_NAME"].astype("string").str.strip()
    frame["GENE_SYMBOL"] = frame["GENE_SYMBOL"].astype("string").str.strip()
    frame["Z_SCORE"] = pd.to_numeric(frame["Z_SCORE"], errors="coerce")
    return frame.dropna(subset=EXPRESSION_COLUMNS)


def build_expression_cache(
    data_dir="data", *, rebuild=False, chunksize=250_000,
    progress: Callable[[str], None] | None = None,
    total_source_rows: int | None = None,
) -> Path:
    """Create/reuse the long Parquet cache with arithmetic-mean duplicates.

    SQLite maintains per sample/gene sums and counts across input chunks, which
    preserves the documented duplicate-resolution policy without loading the
    ~17.5M-row TSV into memory. Pass ``progress=print`` (or another callback)
    to receive chunk and final-writing updates during this long operation.
    ``total_source_rows`` is optional: supplying a verified source row count
    enables percentage progress without an extra full-file scan.
    """
    root = Path(data_dir)
    raw, processed = root / "raw", root / "processed"
    cache = processed / COSMIC_EXPRESSION_PARQUET
    if cache.exists() and not rebuild:
        if progress:
            progress(f"Reusing existing expression cache: {cache}")
        return cache
    source = raw / COSMIC_EXPRESSION_FILE
    if not source.exists():
        download_cosmic_expression(raw)
    processed.mkdir(parents=True, exist_ok=True)
    # Keep the transient SQLite aggregation outside the feature-store directory.
    # Some mounted project volumes permit Parquet output but do not support
    # SQLite's journal/locking semantics, which otherwise raises a disk-I/O or
    # readonly-database error during this memory-safe chunked build.
    descriptor, database_name = tempfile.mkstemp(prefix="gdsc-cosmic-", suffix=".sqlite")
    os.close(descriptor)
    database = Path(database_name)
    connection = sqlite3.connect(database)
    try:
        # This database is disposable intermediate state. Disabling durable
        # journaling avoids a filesystem sync for every source chunk; the raw
        # TSV remains intact and the final Parquet is written only after the
        # aggregation succeeds.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("CREATE TABLE expression (sample_id TEXT, sample_name TEXT, gene TEXT, total REAL, count INTEGER, PRIMARY KEY(sample_id, gene))")
        source_rows = 0
        for chunk_number, chunk in enumerate(pd.read_csv(source, sep="\t", compression="gzip", chunksize=chunksize), start=1):
            values = _normalise_expression(chunk)
            source_rows += len(chunk)
            grouped = values.groupby(["COSMIC_SAMPLE_ID", "SAMPLE_NAME", "GENE_SYMBOL"], as_index=False)["Z_SCORE"].agg(["sum", "count"]).reset_index()
            connection.executemany("INSERT INTO expression VALUES (?, ?, ?, ?, ?) ON CONFLICT(sample_id,gene) DO UPDATE SET total=total+excluded.total,count=count+excluded.count", grouped[["COSMIC_SAMPLE_ID", "SAMPLE_NAME", "GENE_SYMBOL", "sum", "count"]].itertuples(index=False, name=None))
            connection.commit()
            if progress:
                if total_source_rows:
                    percent = 100 * source_rows / total_source_rows
                    progress(f"Aggregated expression chunk {chunk_number:,}: {source_rows:,}/{total_source_rows:,} source rows ({percent:.1f}%).")
                else:
                    progress(f"Aggregated expression chunk {chunk_number:,} ({source_rows:,} source rows read).")
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        try:
            if progress:
                progress("Writing the de-duplicated long-format Parquet cache.")
            written_rows = 0
            for chunk in pd.read_sql_query("SELECT sample_id COSMIC_SAMPLE_ID, sample_name SAMPLE_NAME, gene GENE_SYMBOL, total/count Z_SCORE FROM expression", connection, chunksize=chunksize):
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                writer = writer or pq.ParquetWriter(cache, table.schema, compression="zstd")
                writer.write_table(table)
                written_rows += len(chunk)
                if progress:
                    progress(f"Wrote {written_rows:,} expression feature rows to Parquet.")
        finally:
            if writer:
                writer.close()
    finally:
        connection.close()
        database.unlink(missing_ok=True)
    if progress:
        progress(f"Expression cache complete: {cache}")
    return cache


def load_expression_features(data_dir="data", *, cosmic_sample_ids=None, genes=None) -> pd.DataFrame:
    """Read selected features from the current or legacy Parquet feature store.

    New cache builds write a long-format file in ``data/processed``. Earlier
    project runs generated a wide sample-by-gene Parquet matrix in ``data/raw``.
    The compatibility branch reads only requested samples/genes from that file
    and returns the same long-format API, so it never creates a response-by-gene
    merge and callers do not need to relocate a large generated artifact.
    """
    if cosmic_sample_ids is None and genes is None:
        raise ValueError("Specify cosmic_sample_ids and/or genes; unrestricted load is unsafe")
    root = Path(data_dir)
    processed_cache = root / "processed" / COSMIC_EXPRESSION_PARQUET
    legacy_cache = root / "raw" / COSMIC_EXPRESSION_PARQUET
    if not processed_cache.exists() and legacy_cache.exists():
        import pyarrow.parquet as pq
        sample_ids = list(dict.fromkeys(cosmic_sample_ids or []))
        if not sample_ids:
            raise ValueError("Legacy wide cache requires cosmic_sample_ids")
        schema = pq.ParquetFile(legacy_cache).schema_arrow.names
        gene_columns = [column for column in (genes or schema) if column not in {"COSMIC_SAMPLE_ID", "SAMPLE_NAME"}]
        columns = [column for column in ["COSMIC_SAMPLE_ID", "SAMPLE_NAME", *gene_columns] if column in schema]
        table = pq.read_table(
            legacy_cache,
            columns=columns,
            filters=[("COSMIC_SAMPLE_ID", "in", sample_ids)],
        )
        wide = table.to_pandas()
        if wide.empty:
            return pd.DataFrame(columns=EXPRESSION_COLUMNS)
        return wide.melt(
            id_vars=["COSMIC_SAMPLE_ID", "SAMPLE_NAME"],
            var_name="GENE_SYMBOL",
            value_name="Z_SCORE",
        ).dropna(subset=["Z_SCORE"]).reset_index(drop=True)
    cache = build_expression_cache(data_dir)
    import pyarrow.dataset as ds
    predicates = []
    for field, values in (("COSMIC_SAMPLE_ID", cosmic_sample_ids), ("GENE_SYMBOL", genes)):
        if values is not None:
            values = list(dict.fromkeys(values))
            if not values:
                return pd.DataFrame(columns=EXPRESSION_COLUMNS)
            predicates.append(ds.field(field).isin(values))
    predicate = predicates[0]
    for item in predicates[1:]:
        predicate &= item
    return ds.dataset(cache, format="parquet").to_table(filter=predicate).to_pandas()


def build_sample_mapping(gdsc_metadata: pd.DataFrame, data_dir="data") -> tuple[pd.DataFrame, dict[str, object]]:
    """Map GDSC Sample Name to COSMIC_SAMPLE_ID, retaining unmatched rows."""
    required = {"COSMIC_ID", "Sample Name"}
    missing = required - set(gdsc_metadata.columns)
    if missing:
        raise ValueError(f"GDSC metadata is missing mapping columns: {sorted(missing)}")
    samples = pd.read_csv(Path(data_dir) / "raw" / COSMIC_SAMPLE_FILE, sep="\t", compression="gzip", usecols=["COSMIC_SAMPLE_ID", "SAMPLE_NAME"])
    samples["SAMPLE_NAME"] = samples["SAMPLE_NAME"].astype("string").str.strip()
    ambiguous = samples.groupby("SAMPLE_NAME")["COSMIC_SAMPLE_ID"].nunique()
    ambiguous = ambiguous[ambiguous > 1].index.tolist()
    if ambiguous:
        raise ValueError(f"COSMIC SAMPLE_NAME maps to multiple IDs: {ambiguous[:10]}")
    gdsc = gdsc_metadata[["COSMIC_ID", "Sample Name"]].drop_duplicates().copy()
    gdsc["Sample Name"] = gdsc["Sample Name"].astype("string").str.strip()
    gdsc = gdsc[gdsc["Sample Name"].notna() & gdsc["Sample Name"].ne("") & gdsc["Sample Name"].str.upper().ne("TOTAL:")]
    mapping = gdsc.merge(samples.drop_duplicates("SAMPLE_NAME"), left_on="Sample Name", right_on="SAMPLE_NAME", how="left", validate="one_to_one")
    unmatched = mapping.loc[mapping["COSMIC_SAMPLE_ID"].isna(), "Sample Name"].tolist()
    diagnostics = {"gdsc_sample_names": len(gdsc), "cosmic_sample_names": samples["SAMPLE_NAME"].nunique(), "matched": int(mapping["COSMIC_SAMPLE_ID"].notna().sum()), "unmatched": len(unmatched), "unmatched_names": unmatched, "ambiguous_names": ambiguous}
    return mapping[["COSMIC_ID", "COSMIC_SAMPLE_ID"]], diagnostics
