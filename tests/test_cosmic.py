import sys
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
