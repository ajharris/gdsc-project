import sys
import pandas as pd
import pytest

sys.path.insert(0, "src")
from gdsc import preprocessing as prep


@pytest.fixture
def responses():
    return pd.DataFrame({"COSMIC_ID": [1,2,3,4,5,6], "Sample Name": list("abcdef"), "CELL_LINE_NAME": list("ABCDEF"), "TISSUE_OF_ORIGIN": ["lung","lung","lung","breast","lung","lung"], "DRUG_NAME": ["d1","d1","d1","d1","d2","d2"], "AUC": [.1,.2,.3,.4,.5,.6], "LN_IC50": [1,2,3,4,5,6], "CANCER_TYPE": ["x"]*6, "DATASET": ["GDSC2"]*6})


def test_cohort_response_and_drug_eligibility(responses):
    cohort = prep.select_tissue(responses, "LUNG")
    assert len(cohort) == 5 and len(responses) == 6
    assert prep.select_response_metric(cohort, "LN_IC50").tolist() == [1,2,3,5,6]
    eligibility = prep.drug_eligibility(cohort, min_observations_per_drug=3, min_unique_cell_lines_per_drug=3)
    assert eligibility.set_index("DRUG_NAME").loc["d1", "eligible"]
    assert not eligibility.set_index("DRUG_NAME").loc["d2", "eligible"]
    with pytest.raises(ValueError, match="Unknown"):
        prep.select_tissue(responses, "brain")


def test_tissue_drug_and_duplicate_coverage_summaries(responses):
    duplicate = pd.concat([responses, responses.iloc[[0]].assign(DATASET="GDSC1")], ignore_index=True)
    tissues = prep.summarize_tissues(duplicate)
    assert tissues.loc["lung", "CELL_LINES"] == 5
    assert tissues.loc["lung", "RESPONSE_OBSERVATIONS"] == 6
    drugs = prep.summarize_drugs(duplicate)
    d1 = drugs.set_index("DRUG_NAME").loc["d1"]
    assert d1.N_CELL_LINES == 4 and d1.N_OBSERVATIONS == 5
    assert set(d1.DATASETS) == {"GDSC1", "GDSC2"}
    diagnostics = prep.response_duplicate_diagnostics(duplicate)
    assert diagnostics["n_duplicated_drug_cell_line_pairs"] == 1
    assert diagnostics["max_records_per_pair"] == 2
    coverage = prep.drug_coverage_distribution(drugs, thresholds=(2, 4))
    assert coverage["threshold_counts"] == {2: 2, 4: 1}


def test_missingness_filter_and_training_only_transformer():
    X = pd.DataFrame({"variable": [1., 2., None], "constant": [1., 1., 1.], "empty": [None]*3})
    report = prep.analyze_expression_missingness(X)
    assert report["all_missing_genes"] == ["empty"]
    assert prep.filter_expression_features(X, max_gene_missing_fraction=.5).columns.tolist() == ["variable"]
    transformer = prep.build_preprocessor(scaling=True)
    transformer.fit(X[["variable"]].iloc[:2])
    assert transformer.named_steps["imputer"].statistics_.tolist() == [1.5]


def test_drug_dataset_is_targeted_and_metadata_is_not_features(responses, monkeypatch):
    monkeypatch.setattr(prep, "build_sample_mapping", lambda frame, data_dir: (pd.DataFrame({"COSMIC_ID": frame.COSMIC_ID, "COSMIC_SAMPLE_ID": ["s1","s2","s3"]}), {"matched": 3}))
    monkeypatch.setattr(prep, "load_expression_features", lambda *args, **kwargs: pd.DataFrame({"COSMIC_SAMPLE_ID": ["s1","s1","s2","s2","s3","s3"], "SAMPLE_NAME": ["a"]*6, "GENE_SYMBOL": ["g1","g2"]*3, "Z_SCORE": [1.,2.,3.,4.,5.,6.]}))
    dataset = prep.build_drug_dataset(responses, drug_name="d1", tissue_of_origin="lung", response_metric="AUC", genes=["g1","g2"])
    assert dataset.X.shape == (3, 2)
    assert dataset.y.tolist() == [.1,.2,.3]
    assert "DRUG_NAME" in dataset.metadata and "DRUG_NAME" not in dataset.X


def test_grouped_split_has_no_cell_line_overlap():
    metadata = pd.DataFrame({"COSMIC_ID": list(range(20))})
    dataset = prep.AnalysisDataset(pd.DataFrame({"g": range(20)}), pd.Series(range(20)), metadata, {})
    splits = prep.split_by_cell_line(dataset, test_fraction=.2, validation_fraction=.2, random_state=7)
    sets = [set(split.metadata.COSMIC_ID) for split in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
