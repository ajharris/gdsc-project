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


def test_response_metric_selection_validates_supported_columns_and_preserves_missingness(responses):
    with_missing = responses.copy()
    with_missing.loc[0, "AUC"] = None
    assert prep.select_response_metric(with_missing, "AUC").isna().sum() == 1
    assert prep.select_response_metric(with_missing, "LN_IC50").tolist() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="metric"):
        prep.select_response_metric(with_missing, "IC50")
    with pytest.raises(ValueError, match="does not contain"):
        prep.select_response_metric(with_missing.drop(columns="AUC"), "AUC")


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


def test_drug_identity_ambiguities_are_reported(responses):
    ambiguous = responses.copy()
    ambiguous["DRUG_ID"] = [10, 10, 10, 20, 20, 20]
    ambiguous.loc[1, "DRUG_ID"] = 99
    ambiguous.loc[2, "DRUG_NAME"] = "other-name"
    result = prep.drug_identity_diagnostics(ambiguous)
    assert result["name_to_multiple_ids"].set_index("DRUG_NAME").loc["d1", "N_DRUG_IDS"] == 3
    assert result["id_to_multiple_names"].set_index("DRUG_ID").loc[10, "N_DRUG_NAMES"] == 2


def test_eligibility_uses_unique_cell_lines_and_does_not_mutate_summary():
    summary = pd.DataFrame({"DRUG_NAME": ["many-rows", "enough-lines"], "N_OBSERVATIONS": [100, 3], "N_CELL_LINES": [2, 3]})
    original = summary.copy()
    assert prep.filter_eligible_drugs(summary, min_unique_cell_lines=3)["DRUG_NAME"].tolist() == ["enough-lines"]
    pd.testing.assert_frame_equal(summary, original)
    assert prep.filter_eligible_drugs(summary, min_unique_cell_lines=4).empty
    with pytest.raises(ValueError, match="at least one"):
        prep.filter_eligible_drugs(summary, min_unique_cell_lines=0)


def test_eligibility_report_has_exact_threshold_boundaries_and_counts():
    summary = pd.DataFrame(
        {
            "DRUG_NAME": ["below", "exact", "above"],
            "N_CELL_LINES": [74, 75, 76],
            "N_OBSERVATIONS": [999, 75, 76],
        }
    ).set_index("DRUG_NAME", drop=False)
    report = prep.eligibility_report(summary, min_unique_cell_lines=75)
    assert report["threshold"] == 75
    assert report["total_drugs"] == 3
    assert report["eligible_drugs"].DRUG_NAME.tolist() == ["exact", "above"]
    assert report["ineligible_drugs"].DRUG_NAME.tolist() == ["below"]


def test_cohort_coverage_report_centralizes_notebook_diagnostics(responses):
    report = prep.analyze_cohort_drug_coverage(
        responses.assign(DRUG_ID=[10, 10, 10, 20, 20, 20]),
        tissue_of_origin="lung",
        thresholds=(2, 3),
    )
    assert len(report["cohort"]) == 5
    assert report["drug_summary"].index.tolist() == ["d1", "d2"]
    assert report["coverage_diagnostics"]["threshold_counts"] == {2: 2, 3: 1}
    assert report["eligibility_decision_table"].to_dict("records") == [
        {"minimum_unique_cell_lines": 2, "eligible_drugs": 2},
        {"minimum_unique_cell_lines": 3, "eligible_drugs": 1},
    ]
    assert report["duplicate_diagnostics"]["n_duplicated_drug_cell_line_pairs"] == 0


def test_initial_drug_selection_uses_coverage_then_lowest_stable_id():
    summary = pd.DataFrame(
        {
            "DRUG_NAME": ["below", "higher-id", "lower-id"],
            "DRUG_ID": [(1,), (20,), (10,)],
            "N_CELL_LINES": [74, 75, 75],
            "N_OBSERVATIONS": [1000, 75, 75],
        }
    )
    selected = prep.select_initial_drug(summary, min_unique_cell_lines=75)
    assert selected["DRUG_NAME"] == "lower-id"
    assert selected["DRUG_ID"] == 10
    assert prep.eligibility_report(summary, min_unique_cell_lines=75)["total_drugs"] == 3
    with pytest.raises(ValueError, match="No drugs"):
        prep.select_initial_drug(summary, min_unique_cell_lines=76)
    with pytest.raises(ValueError, match="requires DRUG_ID"):
        prep.select_initial_drug(summary.drop(columns="DRUG_ID"), min_unique_cell_lines=75)


def test_response_dataset_selection_uses_coverage_and_rejects_internal_duplicates(responses):
    records = responses.loc[
        responses.DRUG_NAME.eq("d1") & responses.TISSUE_OF_ORIGIN.eq("lung")
    ].copy()
    records["DRUG_ID"] = 10
    records["DATASET"] = ["GDSC1", "GDSC2", "GDSC2"]
    records = pd.concat([records, records.iloc[[0]].assign(COSMIC_ID=9, DATASET="GDSC2")], ignore_index=True)
    result = prep.select_response_dataset(records, tissue_of_origin="lung", drug_name="d1")
    assert result["selected_dataset"] == "GDSC2"
    assert result["selected_drug_id"] == 10
    assert result["response_cohort"].COSMIC_ID.nunique() == 3
    duplicated = pd.concat([records, records.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="Within-dataset"):
        prep.select_response_dataset(duplicated, tissue_of_origin="lung", drug_name="d1")


def test_response_dataset_tie_prefers_gdsc1_and_counts_missing_target(responses):
    records = responses.loc[
        responses.DRUG_NAME.eq("d1") & responses.TISSUE_OF_ORIGIN.eq("lung")
    ].copy()
    records["DRUG_ID"] = 10
    records["DATASET"] = ["GDSC1", "GDSC1", "GDSC2"]
    records = pd.concat(
        [records, records.iloc[[2]].assign(COSMIC_ID=9, DATASET="GDSC2", AUC=None)],
        ignore_index=True,
    )
    result = prep.select_response_dataset(records, tissue_of_origin="lung", drug_name="d1")
    assert result["selected_dataset"] == "GDSC1"
    assert result["n_excluded_response_rows"] == 0
    assert result["dataset_coverage"].set_index("DATASET").loc["GDSC2", "N_MISSING_RESPONSE"] == 1
    with pytest.raises(ValueError, match="response_metric"):
        prep.select_response_dataset(records, tissue_of_origin="lung", drug_name="d1", response_metric="IC50")


def test_initial_response_cohort_carries_one_screen_one_drug_and_audit_counts(responses):
    data = responses.loc[responses.TISSUE_OF_ORIGIN.eq("lung")].copy()
    data["DRUG_ID"] = [20, 20, 20, 30, 30]
    data["DATASET"] = ["GDSC2"] * 5
    result = prep.build_initial_response_cohort(
        data, tissue_of_origin="lung", min_unique_cell_lines=3, response_metric="AUC"
    )
    final = result["response_cohort"]
    assert result["selected_drug"]["DRUG_NAME"] == "d1"
    assert result["selected_dataset"] == "GDSC2"
    assert final.COSMIC_ID.nunique() == len(final) == 3
    assert final.DRUG_ID.nunique() == final.DATASET.nunique() == 1
    assert result["n_excluded_response_rows"] == 0


def test_missingness_filter_and_training_only_transformer():
    X = pd.DataFrame({"variable": [1., 2., None], "constant": [1., 1., 1.], "empty": [None]*3})
    report = prep.analyze_expression_missingness(X)
    assert report["all_missing_genes"] == ["empty"]
    assert prep.filter_expression_features(X, max_gene_missing_fraction=.5).columns.tolist() == ["variable"]
    transformer = prep.build_preprocessor(scaling=True)
    transformer.fit(X[["variable"]].iloc[:2])
    assert transformer.named_steps["imputer"].statistics_.tolist() == [1.5]
    assert report["n_genes"] == 3 and report["n_cell_lines"] == 3
    assert report["overall_missing_fraction"] == pytest.approx(4 / 9)


def test_expression_dataset_is_targeted_aligned_and_excludes_metadata(responses, monkeypatch):
    cohort = responses.loc[responses.DRUG_NAME.eq("d1") & responses.TISSUE_OF_ORIGIN.eq("lung")].copy()
    cohort["DRUG_ID"] = 10
    monkeypatch.setattr(prep, "build_sample_mapping", lambda frame, data_dir: (
        pd.DataFrame({"COSMIC_ID": [1, 2, 3], "COSMIC_SAMPLE_ID": ["s1", "s2", "s3"]}),
        {"matched": 3},
    ))
    calls = []
    def features(_data_dir, *, cosmic_sample_ids, genes):
        calls.append((cosmic_sample_ids, genes))
        return pd.DataFrame({"COSMIC_SAMPLE_ID": ["s1", "s1", "s2", "s2", "s3", "s3"], "GENE_SYMBOL": ["variable", "constant"] * 3, "Z_SCORE": [1., 1., 2., 1., None, 1.]})
    monkeypatch.setattr(prep, "load_expression_features", features)
    dataset = prep.build_expression_dataset(
        cohort, response_metric="AUC", genes=None, max_gene_missing_fraction=.5
    )
    assert calls == [(["s1", "s2", "s3"], None)]
    assert dataset.X.columns.tolist() == ["variable"]
    assert dataset.y.tolist() == [.1, .2, .3]
    assert dataset.metadata.COSMIC_ID.tolist() == [1, 2, 3]
    assert dataset.diagnostics["missingness_before_filtering"]["n_genes"] == 2
    assert "DRUG_ID" not in dataset.X


def test_expression_dataset_reports_unmatched_and_training_transformer_ignores_validation(monkeypatch):
    cohort = pd.DataFrame({
        "COSMIC_ID": [1, 2], "Sample Name": ["a", "b"], "CELL_LINE_NAME": ["A", "B"],
        "DRUG_ID": [10, 10], "DRUG_NAME": ["d", "d"], "DATASET": ["GDSC1", "GDSC1"],
        "TISSUE_OF_ORIGIN": ["lung", "lung"], "AUC": [1.0, 2.0], "LN_IC50": [3.0, 4.0],
    })
    monkeypatch.setattr(prep, "build_sample_mapping", lambda frame, data_dir: (
        pd.DataFrame({"COSMIC_ID": [1], "COSMIC_SAMPLE_ID": ["s1"]}), {"matched": 1}
    ))
    monkeypatch.setattr(prep, "load_expression_features", lambda *args, **kwargs: pd.DataFrame({
        "COSMIC_SAMPLE_ID": ["s1"], "GENE_SYMBOL": ["g"], "Z_SCORE": [None]
    }))
    dataset = prep.build_expression_dataset(cohort, response_metric="AUC", max_gene_missing_fraction=1.0)
    assert dataset.diagnostics["unmatched_ids"] == [2]
    assert dataset.diagnostics["excluded_no_expression_ids"] == [1]
    transformer = prep.build_preprocessor()
    transformer.fit(pd.DataFrame({"g": [1.0, 2.0, None]}))
    assert transformer.named_steps["imputer"].statistics_.tolist() == [1.5]
    assert transformer.transform(pd.DataFrame({"g": [999.0]})).tolist() == [[999.0]]


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
    repeat = prep.split_by_cell_line(dataset, test_fraction=.2, validation_fraction=.2, random_state=7)
    assert repeat["test"].metadata.COSMIC_ID.tolist() == splits["test"].metadata.COSMIC_ID.tolist()
    for split in splits.values():
        assert len(split.X) == len(split.y) == len(split.metadata)
        assert split.X.index.equals(split.y.index) and split.X.index.equals(split.metadata.index)
    with pytest.raises(ValueError, match="less than one"):
        prep.split_by_cell_line(dataset, test_fraction=.8, validation_fraction=.2)
