"""Leakage-safe preparation of GDSC response/genomic analytical tables.

This module intentionally does not download genomic data or fit learned
transformations.  The pinned GDSC 8.4 files currently available in this
project contain responses and assay-availability flags, but no WES, CNA,
expression, or methylation feature matrix.  Callers must therefore provide a
chosen genomic matrix and the identifier it shares with the response table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


DEFAULT_METADATA_COLUMNS = (
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "TISSUE_OF_ORIGIN",
    "CANCER_TYPE",
    "DRUG_ID",
    "DRUG_NAME",
    "DATASET",
)


@dataclass(frozen=True)
class PreprocessedGDSC:
    """Analytical inputs plus an auditable account of preprocessing choices.

    ``X`` has genomic features only. ``y`` is the explicitly selected response
    measure. ``metadata`` remains separate so tissue and drug can be used for
    grouping/evaluation without becoming predictive inputs.
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    info: dict[str, object]


def _validate_fraction(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _summary(metadata: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Count observations and distinct cell lines for a grouping column."""
    if column not in metadata:
        raise ValueError(f"Metadata does not contain {column!r}")
    cell_line_column = "COSMIC_ID" if "COSMIC_ID" in metadata else metadata.columns[0]
    return (
        metadata.groupby(column, dropna=False)
        .agg(observations=(column, "size"), cell_lines=(cell_line_column, "nunique"))
        .sort_values("observations", ascending=False)
        .rename_axis(label)
    )


def summarize_tissues(
    metadata: pd.DataFrame, tissue_column: str = "TISSUE_OF_ORIGIN"
) -> pd.DataFrame:
    """Summarize retained observations and cell lines by tissue of origin."""
    return _summary(metadata, tissue_column, "tissue")


def summarize_drugs(
    metadata: pd.DataFrame, drug_column: str = "DRUG_NAME"
) -> pd.DataFrame:
    """Summarize retained observations and cell lines by drug identity."""
    return _summary(metadata, drug_column, "drug")


def build_training_transformer(
    *,
    imputation: Literal["median", "mean", "most_frequent"] | None = None,
    scaling: Literal["standard", "robust"] | None = None,
):
    """Return an **unfitted** scikit-learn transformer for a later split.

    This helper is intentionally separate from :func:`preprocess_gdsc`: callers
    must call ``fit`` on training features only, then ``transform`` validation
    and test features.  No global imputation or scaling occurs in this module.
    """
    if imputation is None and scaling is None:
        return None
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ImportError(
            "build_training_transformer requires scikit-learn; install the "
            "project's machine-learning dependencies."
        ) from error

    steps = []
    if imputation is not None:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "robust":
        steps.append(("scaler", RobustScaler()))
    return Pipeline(steps)


def preprocess_gdsc(
    response_data: pd.DataFrame,
    genomic_data: pd.DataFrame,
    *,
    response_metric: str = "AUC",
    response_cell_line_id: str = "COSMIC_ID",
    genomic_cell_line_id: str | None = None,
    tissue_column: str = "TISSUE_OF_ORIGIN",
    drug_column: str = "DRUG_NAME",
    metadata_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS,
    feature_columns: list[str] | None = None,
    min_tissue_observations: int = 1,
    min_drug_observations: int = 1,
    min_feature_availability: float = 0.8,
    max_feature_missingness: float | None = None,
    min_feature_variance: float = 0.0,
    missing_data_strategy: Literal["keep", "drop_observations"] = "keep",
) -> PreprocessedGDSC:
    """Join an explicitly supplied genomic matrix to GDSC response records.

    Defaults are transparent starting points, not final scientific choices:
    each retained genomic feature must be present in at least 80% of matched
    observations, and constant features are removed.  Eligibility thresholds
    are evaluated after records without the requested response have been
    explicitly removed.  ``missing_data_strategy='keep'`` is the safe default:
    imputation must be fitted on a future training split with
    :func:`build_training_transformer`.
    """
    if response_metric not in response_data:
        raise ValueError(f"Response data does not contain {response_metric!r}")
    if response_cell_line_id not in response_data:
        raise ValueError(
            f"Response data does not contain identifier {response_cell_line_id!r}"
        )
    genomic_cell_line_id = genomic_cell_line_id or response_cell_line_id
    if genomic_cell_line_id not in genomic_data:
        raise ValueError(
            f"Genomic data does not contain identifier {genomic_cell_line_id!r}; "
            "pass genomic_cell_line_id for the verified shared identifier."
        )
    if tissue_column not in response_data or drug_column not in response_data:
        raise ValueError("Response data must contain the configured tissue and drug columns")
    if min_tissue_observations < 1 or min_drug_observations < 1:
        raise ValueError("Minimum observation thresholds must be at least 1")
    _validate_fraction(min_feature_availability, "min_feature_availability")
    if max_feature_missingness is not None:
        _validate_fraction(max_feature_missingness, "max_feature_missingness")
        min_feature_availability = 1 - max_feature_missingness
    if min_feature_variance < 0:
        raise ValueError("min_feature_variance must be non-negative")
    if missing_data_strategy not in {"keep", "drop_observations"}:
        raise ValueError("missing_data_strategy must be 'keep' or 'drop_observations'")
    if genomic_data[genomic_cell_line_id].duplicated().any():
        raise ValueError("Genomic data must have one row per genomic cell-line identifier")

    # Select genomic columns before the join: response/tissue/drug metadata can
    # never leak into X, even if similarly named columns occur in both tables.
    candidate_features = feature_columns if feature_columns is not None else [
        column for column in genomic_data.columns if column != genomic_cell_line_id
    ]
    missing_features = set(candidate_features) - set(genomic_data.columns)
    if missing_features:
        raise ValueError(f"Genomic data is missing requested features: {sorted(missing_features)}")
    non_numeric = [
        column for column in candidate_features
        if not pd.api.types.is_numeric_dtype(genomic_data[column])
    ]
    if non_numeric:
        raise ValueError(
            "Genomic features must be numeric; encode/select features explicitly: "
            f"{non_numeric}"
        )

    response_columns = list(
        dict.fromkeys(
            [response_cell_line_id, response_metric, tissue_column, drug_column, *metadata_columns]
        )
    )
    response_columns = [column for column in response_columns if column in response_data]
    response = response_data.loc[:, response_columns].copy()
    genomic = genomic_data.loc[:, [genomic_cell_line_id, *candidate_features]].copy()
    if genomic_cell_line_id != response_cell_line_id:
        genomic = genomic.rename(columns={genomic_cell_line_id: response_cell_line_id})

    observations_before_join = len(response)
    matched_cell_lines = set(response[response_cell_line_id].dropna()).intersection(
        genomic[response_cell_line_id].dropna()
    )
    joined = response.merge(genomic, on=response_cell_line_id, how="inner", validate="many_to_one")
    observations_after_join = len(joined)
    joined = joined.loc[joined[response_metric].notna()].copy()
    observations_after_response_filter = len(joined)

    # Feature-level availability is measured only among usable joined records.
    availability = joined[candidate_features].notna().mean() if candidate_features else pd.Series(dtype=float)
    retained_features = availability[availability >= min_feature_availability].index.tolist()
    if retained_features:
        variances = joined[retained_features].var(skipna=True)
        retained_features = variances[variances > min_feature_variance].index.tolist()
    else:
        variances = pd.Series(dtype=float)

    if missing_data_strategy == "drop_observations" and retained_features:
        joined = joined.loc[joined[retained_features].notna().all(axis=1)].copy()

    # Apply eligibility without consulting model performance; repeat until both
    # minimum-count constraints hold after their interaction.
    while not joined.empty:
        tissue_counts = joined[tissue_column].value_counts(dropna=False)
        drug_counts = joined[drug_column].value_counts(dropna=False)
        eligible = joined[tissue_column].map(tissue_counts).ge(min_tissue_observations)
        eligible &= joined[drug_column].map(drug_counts).ge(min_drug_observations)
        if eligible.all():
            break
        joined = joined.loc[eligible].copy()

    retained_metadata = [column for column in metadata_columns if column in joined]
    # Include custom identifier once so grouping tables always retain it.
    if response_cell_line_id not in retained_metadata:
        retained_metadata.insert(0, response_cell_line_id)
    X = joined.loc[:, retained_features].copy()
    y = joined[response_metric].copy().rename("y")
    metadata = joined.loc[:, retained_metadata].copy()
    info = {
        "response_metric": response_metric,
        "response_cell_line_id": response_cell_line_id,
        "observations_before_join": observations_before_join,
        "genomic_cell_lines": int(genomic[response_cell_line_id].nunique()),
        "matched_cell_lines": len(matched_cell_lines),
        "observations_after_join": observations_after_join,
        "observations_after_response_filter": observations_after_response_filter,
        "observations_final": len(X),
        "features_before_filtering": len(candidate_features),
        "features_final": len(retained_features),
        "min_tissue_observations": min_tissue_observations,
        "min_drug_observations": min_drug_observations,
        "min_feature_availability": min_feature_availability,
        "max_feature_missingness": 1 - min_feature_availability,
        "min_feature_variance": min_feature_variance,
        "missing_data_strategy": missing_data_strategy,
        "feature_availability": availability.to_dict(),
        "feature_variance": variances.to_dict(),
    }
    return PreprocessedGDSC(X=X, y=y, metadata=metadata, info=info)
