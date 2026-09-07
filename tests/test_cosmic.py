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
    updates = []
    cache = cosmic.build_expression_cache(cosmic_dir, chunksize=2, progress=updates.append, total_source_rows=4)
    assert cache == cosmic_dir / "processed" / cosmic.COSMIC_EXPRESSION_PARQUET
    result = pd.read_parquet(cache)
    assert set(result.columns) == set(cosmic.EXPRESSION_COLUMNS)
    assert result.loc[(result.COSMIC_SAMPLE_ID == "C1") & (result.GENE_SYMBOL == "TP53"), "Z_SCORE"].iloc[0] == pytest.approx(1.01)
    assert any("Aggregated expression chunk" in update for update in updates)
    assert any("4/4" in update for update in updates)
    assert any("Writing" in update for update in updates)
    assert updates[-1].startswith("Expression cache complete")
    monkeypatch.setattr(cosmic.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw reread")))
    assert cosmic.build_expression_cache(cosmic_dir) == cache


def test_targeted_queries_are_restricted(cosmic_dir):
    assert cosmic.load_expression_features(cosmic_dir, cosmic_sample_ids=["C1"], genes=["EGFR"]).to_dict("records") == [{"COSMIC_SAMPLE_ID": "C1", "SAMPLE_NAME": "Alpha", "GENE_SYMBOL": "EGFR", "Z_SCORE": 2.0}]
    assert set(cosmic.load_expression_features(cosmic_dir, cosmic_sample_ids=["C1", "C1"])["GENE_SYMBOL"]) == {"TP53", "EGFR"}
    assert set(cosmic.load_expression_features(cosmic_dir, genes=["TP53", "TP53"])["COSMIC_SAMPLE_ID"]) == {"C1", "C2"}
    assert cosmic.load_expression_features(cosmic_dir, genes=[]).empty
    with pytest.raises(ValueError, match="unrestricted"):
        cosmic.load_expression_features(cosmic_dir)


def test_legacy_wide_parquet_in_raw_is_read_at_its_generated_location(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "COSMIC_SAMPLE_ID": ["C1", "C2"], "SAMPLE_NAME": ["Alpha", "Beta"],
        "TP53": [1.0, 2.0], "EGFR": [3.0, 4.0],
    }).to_parquet(raw / cosmic.COSMIC_EXPRESSION_PARQUET, index=False)
    result = cosmic.load_expression_features(tmp_path, cosmic_sample_ids=["C2"], genes=["EGFR"])
    assert result.to_dict("records") == [{"COSMIC_SAMPLE_ID": "C2", "SAMPLE_NAME": "Beta", "GENE_SYMBOL": "EGFR", "Z_SCORE": 4.0}]


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


@pytest.mark.parametrize("cache_exists", [False, True])
def test_mapping_without_sample_file_builds_or_reuses_expression(cosmic_dir, monkeypatch, cache_exists):
    (cosmic_dir / "raw" / cosmic.COSMIC_SAMPLE_FILE).unlink()
    if cache_exists:
        cosmic.build_expression_cache(cosmic_dir)
        (cosmic_dir / "raw" / cosmic.COSMIC_EXPRESSION_FILE).unlink()
    monkeypatch.setattr(cosmic, "_download_file", lambda *args: pytest.fail("downloaded"))
    metadata = pd.DataFrame({"COSMIC_ID": [1, 2, 3], "Sample Name": ["Alpha", "Beta", "Missing"]})
    mapping, diagnostics = cosmic.build_sample_mapping(metadata, cosmic_dir)
    assert mapping["COSMIC_SAMPLE_ID"].iloc[:2].tolist() == ["C1", "C2"]
    assert diagnostics["matched"] == 2
    assert diagnostics["unmatched_names"] == ["Missing"]
    assert (cosmic_dir / "processed" / cosmic.COSMIC_EXPRESSION_PARQUET).exists()


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


def _archive_bytes():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(cosmic.COSMIC_EXPRESSION_FILE)
        member.size = 7
        archive.addfile(member, io.BytesIO(b"payload"))
    return stream.getvalue()


def test_download_follows_metadata_and_repairs_cached_json(tmp_path, monkeypatch):
    archive = tmp_path / cosmic.COSMIC_EXPRESSION_ARCHIVE
    archive.write_text('{"url": "https://example.test/expired"}')
    monkeypatch.setenv("COSMIC_LINK", "https://example.test/api")
    calls = []
    responses = [b'{"url": "https://example.test/signed"}', _archive_bytes()]

    def open_response(request):
        calls.append(request)
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr(cosmic, "urlopen", open_response)
    result = cosmic.download_cosmic_expression(tmp_path)
    assert result["expression"].read_bytes() == b"payload"
    assert calls[1] == "https://example.test/signed"
    assert tarfile.is_tarfile(archive)
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("payload", [b"<html>Login required</html>", b'{"error": "denied"}', b'{"url": "file:///etc/passwd"}'])
def test_invalid_download_never_publishes_archive(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(cosmic, "urlopen", lambda request: io.BytesIO(payload))
    destination = tmp_path / "expression.tar"
    with pytest.raises(RuntimeError, match="COSMIC"):
        cosmic._download_file("https://example.test/api", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_interrupted_download_preserves_existing_file(tmp_path, monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(size)

    monkeypatch.setattr(cosmic, "urlopen", lambda request: Interrupted(_archive_bytes()))
    destination = tmp_path / "expression.tar"
    destination.write_bytes(b"existing")
    with pytest.raises(OSError, match="interrupted"):
        cosmic._download_file("https://example.test/archive", destination)
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.part"))


def test_project_env_takes_precedence_over_example(tmp_path):
    import os
    import subprocess
    source = Path(cosmic.__file__).read_text()
    # Exercise module initialization in a fresh process with isolated test config.
    defaults = Path(cosmic.PROJECT_ROOT / '.env.example').read_text()
    (tmp_path / '.env.example').write_text(defaults)
    (tmp_path / '.env').write_text('COSMIC_AUTHORIZATION=fixture-token\n')
    source = source.replace('PROJECT_ROOT = Path(__file__).resolve().parents[2]',
                            f'PROJECT_ROOT = Path({str(tmp_path)!r})')
    module = tmp_path / 'cosmic_config_test.py'
    module.write_text(source + '\nassert _cosmic_request().get_header("Authorization") == "Basic fixture-token"\n')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('COSMIC_')}
    subprocess.run([sys.executable, str(module)], env=environment, check=True, capture_output=True)


def test_empty_signed_link_falls_back_to_endpoint(monkeypatch):
    monkeypatch.setenv('COSMIC_LINK', ' ')
    monkeypatch.setenv('COSMIC_AUTHORIZATION', 'test-token')
    assert cosmic._cosmic_request().full_url == cosmic.COSMIC_EXPRESSION_URL


@pytest.mark.parametrize("location", ["project", "working_directory"])
def test_request_loads_env_added_after_import(tmp_path, monkeypatch, location):
    project = tmp_path / "project"
    working = tmp_path / "notebooks"
    project.mkdir()
    working.mkdir()
    monkeypatch.setattr(cosmic, "PROJECT_ROOT", project)
    monkeypatch.chdir(working)
    monkeypatch.setenv("COSMIC_AUTHORIZATION", " ")
    monkeypatch.setenv("COSMIC_LINK", "")
    directory = project if location == "project" else working
    (directory / ".env").write_text('\ufeffCOSMIC_AUTHORIZATION=late-test-token\n')
    assert cosmic._cosmic_request().get_header("Authorization") == "Basic late-test-token"


def test_request_preserves_existing_runtime_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(cosmic, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("COSMIC_AUTHORIZATION", "runtime-test-token")
    monkeypatch.setenv("COSMIC_LINK", "")
    (tmp_path / ".env").write_text('COSMIC_AUTHORIZATION=file-test-token\n')
    assert cosmic._cosmic_request().get_header("Authorization") == "Basic runtime-test-token"
