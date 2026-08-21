"""Synthetic tests for preprocessing; no downloaded GDSC data are required."""

import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")

from gdsc.preprocessing import (
    build_training_transformer,
    preprocess_gdsc,
    summarize_drugs,
    summarize_tissues,
)


@pytest.fixture
def response_data():
    return pd.DataFrame(
        {
            "COSMIC_ID": [1, 1, 2, 3, 4],
            "CELL_LINE_NAME": ["one", "one", "two", "three", "unmatched"],
            "TISSUE_OF_ORIGIN": ["lung", "lung", "breast", "lung", "breast"],
            "CANCER_TYPE": ["LUAD", "LUAD", "BRCA", "LUAD", "BRCA"],
            "DRUG_ID": [10, 11, 10, 10, 12],
            "DRUG_NAME": ["drug-a", "drug-b", "drug-a", "drug-a", "drug-c"],
            "AUC": [0.1, 0.2, 0.3, None, 0.5],
            "LN_IC50": [1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )


@pytest.fixture
def genomic_data():
    return pd.DataFrame(
        {
            "COSMIC_ID": [1, 2, 3],
            "GENE_A": [0.0, 1.0, 1.0],
            "GENE_B": [4.0, None, 6.0],
            "CONSTANT": [1.0, 1.0, 1.0],
        }
    )


def test_join_response_selection_and_metadata_exclusion(response_data, genomic_data):
    result = preprocess_gdsc(response_data, genomic_data, response_metric="LN_IC50")

    assert result.info["observations_before_join"] == 5
    assert result.info["genomic_cell_lines"] == 3
    assert result.info["matched_cell_lines"] == 3
    assert result.info["observations_after_join"] == 4
    assert result.y.tolist() == [1.1, 1.2, 1.3, 1.4]
    assert set(result.metadata.columns) >= {
        "COSMIC_ID", "TISSUE_OF_ORIGIN", "CANCER_TYPE", "DRUG_NAME"
    }
    assert list(result.X.columns) == ["GENE_A"]
    assert not set(result.metadata.columns).intersection(result.X.columns)
    assert "TISSUE_OF_ORIGIN" not in result.X
    assert "DRUG_NAME" not in result.X


def test_missing_data_is_kept_by_default_or_explicitly_dropped(response_data, genomic_data):
    kept = preprocess_gdsc(
        response_data,
        genomic_data,
        response_metric="LN_IC50",
        min_feature_availability=0.5,
        min_feature_variance=0,
    )
    dropped = preprocess_gdsc(
        response_data,
        genomic_data,
        response_metric="LN_IC50",
        min_feature_availability=0.5,
        missing_data_strategy="drop_observations",
    )

    assert "GENE_B" in kept.X
    assert kept.X["GENE_B"].isna().sum() == 1
    assert len(dropped.X) == len(kept.X) - 1
    assert dropped.X.isna().sum().sum() == 0


def test_filters_features_and_applies_configurable_eligibility(response_data, genomic_data):
    response_data = response_data.copy()
    response_data.loc[3, "AUC"] = 0.4
    result = preprocess_gdsc(
        response_data,
        genomic_data,
        min_feature_availability=0.0,
        min_feature_variance=0.1,
        min_tissue_observations=2,
        min_drug_observations=2,
    )

    assert set(result.X.columns) == {"GENE_A", "GENE_B"}
    assert "CONSTANT" not in result.X
    assert result.metadata["TISSUE_OF_ORIGIN"].tolist() == ["lung", "lung"]
    assert result.info["min_tissue_observations"] == 2
    assert result.info["min_drug_observations"] == 2


def test_summaries_count_observations_and_cell_lines(response_data, genomic_data):
    result = preprocess_gdsc(response_data, genomic_data)
    tissues = summarize_tissues(result.metadata)
    drugs = summarize_drugs(result.metadata)

    assert tissues.loc["lung", "observations"] == 2
    assert tissues.loc["lung", "cell_lines"] == 1
    assert drugs.loc["drug-a", "observations"] == 2


def test_unmatched_join_and_invalid_response_are_explicit(response_data, genomic_data):
    unmatched = genomic_data.assign(COSMIC_ID=[99, 98, 97])
    result = preprocess_gdsc(response_data, unmatched)

    assert result.X.empty
    assert result.y.empty
    assert result.info["matched_cell_lines"] == 0
    with pytest.raises(ValueError, match="NOT_A_RESPONSE"):
        preprocess_gdsc(response_data, genomic_data, response_metric="NOT_A_RESPONSE")


def test_custom_identifier_and_missingness_alias(response_data, genomic_data):
    genomic = genomic_data.rename(columns={"COSMIC_ID": "MODEL_ID"})
    result = preprocess_gdsc(
        response_data,
        genomic,
        response_metric="LN_IC50",
        genomic_cell_line_id="MODEL_ID",
        max_feature_missingness=0.5,
    )

    assert result.info["max_feature_missingness"] == 0.5
    assert "GENE_B" in result.X


def test_training_transformer_is_unfitted_until_the_training_split_is_supplied():
    transformer = build_training_transformer(imputation="median", scaling="standard")

    # The helper returns configuration only; fitting is a later split-specific step.
    assert not hasattr(transformer.named_steps["imputer"], "statistics_")
    transformer.fit(pd.DataFrame({"GENE": [1.0, 3.0, None]}))
    assert transformer.named_steps["imputer"].statistics_.tolist() == [2.0]
