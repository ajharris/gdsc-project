import sys
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "src")
from gdsc import cosmic


def _expression(path):
    pd.DataFrame({
        "COSMIC_SAMPLE_ID": ["C1", "C1", "C1", "C2"],
        "SAMPLE_NAME": [" Alpha ", " Alpha ", " Alpha ", "Beta"],
        "COSMIC_GENE_ID": ["G1"] * 4,
        "GENE_SYMBOL": ["TP53", "TP53", "EGFR", "TP53"],
        "REGULATION": ["normal"] * 4,
        "Z_SCORE": [1.0, 1.02, 2.0, -1.0],
        "COSMIC_STUDY_ID": ["COSU619", "COSU3", "COSU619", "COSU619"],
    }).to_csv(path, sep="\t", index=False, compression="gzip")


def _samples(path):
    pd.DataFrame({"COSMIC_SAMPLE_ID": ["C1", "C2"], "SAMPLE_NAME": ["Alpha", "Beta"]}).to_csv(path, sep="\t", index=False, compression="gzip")


@pytest.fixture
def cosmic_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _expression(raw / cosmic.COSMIC_EXPRESSION_FILE)
    _samples(raw / cosmic.COSMIC_SAMPLE_FILE)
    return tmp_path


def test_cache_is_long_deduplicated_and_reused(cosmic_dir, monkeypatch):
    cache = cosmic.build_expression_cache(cosmic_dir, chunksize=2)
    assert cache == cosmic_dir / "processed" / cosmic.COSMIC_EXPRESSION_PARQUET
    result = pd.read_parquet(cache)
    assert set(result.columns) == set(cosmic.EXPRESSION_COLUMNS)
    assert result.loc[(result.COSMIC_SAMPLE_ID == "C1") & (result.GENE_SYMBOL == "TP53"), "Z_SCORE"].iloc[0] == pytest.approx(1.01)
    monkeypatch.setattr(cosmic.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw reread")))
    assert cosmic.build_expression_cache(cosmic_dir) == cache


def test_targeted_queries_are_restricted(cosmic_dir):
    assert cosmic.load_expression_features(cosmic_dir, cosmic_sample_ids=["C1"], genes=["EGFR"]).to_dict("records") == [{"COSMIC_SAMPLE_ID": "C1", "SAMPLE_NAME": "Alpha", "GENE_SYMBOL": "EGFR", "Z_SCORE": 2.0}]
    assert set(cosmic.load_expression_features(cosmic_dir, cosmic_sample_ids=["C1", "C1"])["GENE_SYMBOL"]) == {"TP53", "EGFR"}
    assert set(cosmic.load_expression_features(cosmic_dir, genes=["TP53", "TP53"])["COSMIC_SAMPLE_ID"]) == {"C1", "C2"}
    assert cosmic.load_expression_features(cosmic_dir, genes=[]).empty
    with pytest.raises(ValueError, match="unrestricted"):
        cosmic.load_expression_features(cosmic_dir)


def test_mapping_normalizes_names_excludes_total_and_reports_unmatched(cosmic_dir):
    metadata = pd.DataFrame({"COSMIC_ID": [1, 2, 3], "Sample Name": [" Alpha ", "TOTAL:", "Missing"]})
    mapping, diagnostics = cosmic.build_sample_mapping(metadata, cosmic_dir)
    assert mapping.loc[mapping["COSMIC_ID"] == 1, "COSMIC_SAMPLE_ID"].iloc[0] == "C1"
    assert mapping.loc[mapping["COSMIC_ID"] == 3, "COSMIC_SAMPLE_ID"].isna().all()
    assert diagnostics["matched"] == 1
    assert diagnostics["unmatched_names"] == ["Missing"]


def test_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="GENE_SYMBOL"):
        cosmic._normalise_expression(pd.DataFrame({"COSMIC_SAMPLE_ID": ["C1"]}))


def test_request_uses_signed_link_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("COSMIC_LINK", "https://example.test/signed")
    monkeypatch.delenv("COSMIC_AUTHORIZATION", raising=False)
    assert cosmic._cosmic_request().full_url == "https://example.test/signed"
    monkeypatch.setenv("COSMIC_AUTHORIZATION", "redacted-token")
    request = cosmic._cosmic_request()
    assert request.full_url == "https://example.test/signed"
    assert request.get_header("Authorization") == "Basic redacted-token"


def test_download_reuses_archive_and_extracts_exact_member(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = raw / cosmic.COSMIC_EXPRESSION_ARCHIVE
    payload = b"compressed-expression"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(cosmic.COSMIC_EXPRESSION_FILE)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(cosmic, "_download_file", lambda *args: pytest.fail("downloaded"))
    result = cosmic.download_cosmic_expression(raw)
    assert result["expression"].read_bytes() == payload
    assert cosmic.download_cosmic_expression(raw) == result


def test_extract_rejects_missing_or_duplicate_members(tmp_path):
    missing = tmp_path / "missing.tar"
    with tarfile.open(missing, "w") as tar:
        info = tarfile.TarInfo("other.tsv.gz")
        info.size = 0
        tar.addfile(info)
    with pytest.raises(ValueError, match="found 0"):
        cosmic._extract_member(missing, tmp_path, cosmic.COSMIC_EXPRESSION_FILE)
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "w") as tar:
        for directory in ("a", "b"):
            info = tarfile.TarInfo(f"{directory}/{cosmic.COSMIC_EXPRESSION_FILE}")
            info.size = 0
            tar.addfile(info)
    with pytest.raises(ValueError, match="found 2"):
        cosmic._extract_member(duplicate, tmp_path, cosmic.COSMIC_EXPRESSION_FILE)


def test_mapping_rejects_ambiguous_cosmic_sample_name(cosmic_dir):
    _samples(cosmic_dir / "raw" / cosmic.COSMIC_SAMPLE_FILE)
    pd.DataFrame({"COSMIC_SAMPLE_ID": ["C1", "C9"], "SAMPLE_NAME": ["Alpha", "Alpha"]}).to_csv(
        cosmic_dir / "raw" / cosmic.COSMIC_SAMPLE_FILE, sep="\t", index=False, compression="gzip"
    )
    with pytest.raises(ValueError, match="multiple IDs"):
        cosmic.build_sample_mapping(pd.DataFrame({"COSMIC_ID": [1], "Sample Name": ["Alpha"]}), cosmic_dir)
