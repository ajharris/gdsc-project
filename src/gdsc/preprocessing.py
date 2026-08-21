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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from gdsc.cosmic import build_sample_mapping, load_expression_features


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


@dataclass(frozen=True)
class AnalysisDataset:
    """One-drug, one-row-per-cell-line modelling dataset.

    Expression is queried only for the selected mapped cell lines and requested
    genes. Metadata deliberately remains outside ``X``.
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    diagnostics: dict[str, object]


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


def select_tissue(data: pd.DataFrame, tissue_of_origin: str) -> pd.DataFrame:
    """Return a case-insensitive tissue cohort without mutating ``data``."""
    if "TISSUE_OF_ORIGIN" not in data:
        raise ValueError("Data does not contain TISSUE_OF_ORIGIN")
    cohort = data.loc[data["TISSUE_OF_ORIGIN"].astype("string").str.casefold().eq(tissue_of_origin.casefold())].copy()
    if cohort.empty:
        available = sorted(data["TISSUE_OF_ORIGIN"].dropna().astype(str).unique())
        raise ValueError(f"Unknown or empty tissue {tissue_of_origin!r}; available: {available}")
    return cohort


def select_response_metric(data: pd.DataFrame, metric: str = "AUC") -> pd.Series:
    """Return the explicitly requested non-missing GDSC response target."""
    if metric not in {"AUC", "LN_IC50"}:
        raise ValueError("metric must be 'AUC' or 'LN_IC50'")
    if metric not in data:
        raise ValueError(f"Data does not contain response metric {metric!r}")
    return data[metric].copy().rename("y")


def drug_eligibility(
    cohort: pd.DataFrame, *, min_observations_per_drug: int, min_unique_cell_lines_per_drug: int
) -> pd.DataFrame:
    """Summarize and flag drugs using counts within the supplied cohort only."""
    if min_observations_per_drug < 1 or min_unique_cell_lines_per_drug < 1:
        raise ValueError("Eligibility thresholds must be at least one")
    summary = cohort.groupby("DRUG_NAME", dropna=False).agg(n_observations=("DRUG_NAME", "size"), n_cell_lines=("COSMIC_ID", "nunique")).reset_index()
    summary["eligible"] = summary["n_observations"].ge(min_observations_per_drug) & summary["n_cell_lines"].ge(min_unique_cell_lines_per_drug)
    return summary.sort_values("n_observations", ascending=False).reset_index(drop=True)


def analyze_expression_missingness(X: pd.DataFrame) -> dict[str, object]:
    """Return feature and cell-line missingness before any imputation."""
    return {"gene_missing_fraction": X.isna().mean(), "cell_line_missing_fraction": X.isna().mean(axis=1), "total_missing": int(X.isna().sum().sum()), "all_missing_genes": X.columns[X.isna().all()].tolist(), "empty_cell_lines": X.index[X.isna().all(axis=1)].tolist()}


def filter_expression_features(X: pd.DataFrame, *, max_gene_missing_fraction: float = 0.2, max_cell_line_missing_fraction: float = 1.0, min_variance: float = 0.0) -> pd.DataFrame:
    """Remove overly missing rows/features and zero/near-zero variance genes.

    Defaults are transparent starting points; numeric filtering should be fitted
    on training data in a production evaluation when cohort-wide leakage is a concern.
    """
    _validate_fraction(max_gene_missing_fraction, "max_gene_missing_fraction")
    _validate_fraction(max_cell_line_missing_fraction, "max_cell_line_missing_fraction")
    filtered = X.loc[X.isna().mean(axis=1).le(max_cell_line_missing_fraction)].copy()
    filtered = filtered.loc[:, filtered.isna().mean().le(max_gene_missing_fraction)]
    return filtered.loc[:, filtered.var(skipna=True).gt(min_variance)]


def build_drug_dataset(
    responses: pd.DataFrame, *, drug_name: str, tissue_of_origin: str, response_metric: str,
    genes: list[str], data_dir="data", min_observations_per_drug: int = 1, min_unique_cell_lines_per_drug: int = 1,
) -> AnalysisDataset:
    """Build a bounded, drug-specific dataset; never a full response/gene join."""
    cohort = select_tissue(responses, tissue_of_origin)
    eligibility = drug_eligibility(cohort, min_observations_per_drug=min_observations_per_drug, min_unique_cell_lines_per_drug=min_unique_cell_lines_per_drug)
    row = eligibility.loc[eligibility["DRUG_NAME"].eq(drug_name)]
    if row.empty or not bool(row["eligible"].iloc[0]):
        raise ValueError(f"Drug {drug_name!r} is not eligible in tissue {tissue_of_origin!r}")
    selected = cohort.loc[cohort["DRUG_NAME"].eq(drug_name) & cohort[response_metric].notna()].copy()
    if selected.duplicated("COSMIC_ID").any():
        raise ValueError("Drug-specific dataset requires one response per COSMIC_ID")
    mapping, mapping_info = build_sample_mapping(selected[["COSMIC_ID", "Sample Name"]], data_dir)
    selected = selected.merge(mapping, on="COSMIC_ID", how="left", validate="one_to_one")
    mapped = selected.loc[selected["COSMIC_SAMPLE_ID"].notna()].copy()
    expression = load_expression_features(data_dir, cosmic_sample_ids=mapped["COSMIC_SAMPLE_ID"].tolist(), genes=genes)
    wide = expression.pivot(index="COSMIC_SAMPLE_ID", columns="GENE_SYMBOL", values="Z_SCORE").reindex(mapped["COSMIC_SAMPLE_ID"])
    wide.index = mapped.index
    available = wide.notna().any(axis=1)
    metadata_columns = [column for column in ("COSMIC_ID", "COSMIC_SAMPLE_ID", "CELL_LINE_NAME", "DRUG_NAME", "TISSUE_OF_ORIGIN", "CANCER_TYPE", "DATASET") if column in mapped]
    metadata = mapped.loc[available, metadata_columns].copy()
    X = wide.loc[available].copy()
    y = mapped.loc[available, response_metric].copy().rename("y")
    diagnostics = {"drug_eligibility": eligibility, "mapping": mapping_info, "n_response_cell_lines": int(selected["COSMIC_ID"].nunique()), "n_mapped_to_cosmic": len(mapped), "n_with_expression": int(available.sum()), "n_excluded": int((~available).sum()), "excluded_ids": mapped.loc[~available, "COSMIC_ID"].tolist()}
    return AnalysisDataset(X=X, y=y, metadata=metadata, diagnostics=diagnostics)


def split_by_cell_line(dataset: AnalysisDataset, *, test_fraction: float = 0.2, validation_fraction: float = 0.2, random_state: int = 42) -> dict[str, AnalysisDataset]:
    """Create train/validation/test splits with no COSMIC_ID overlap."""
    _validate_fraction(test_fraction, "test_fraction"); _validate_fraction(validation_fraction, "validation_fraction")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than one")
    groups = dataset.metadata["COSMIC_ID"]
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state).split(dataset.X, groups=groups))
    validation_relative = validation_fraction / (1 - test_fraction)
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=validation_relative, random_state=random_state).split(dataset.X.iloc[train_val], groups=groups.iloc[train_val]))
    def subset(indices):
        return AnalysisDataset(dataset.X.iloc[indices].copy(), dataset.y.iloc[indices].copy(), dataset.metadata.iloc[indices].copy(), dataset.diagnostics)
    return {"train": subset(train_val[train]), "validation": subset(train_val[validation]), "test": subset(test)}


def build_preprocessor(*, imputation: str = "median", scaling: bool = False, variance_threshold: float = 0.0) -> Pipeline:
    """Return an unfitted pipeline; callers fit it on the training split only."""
    steps = [("imputer", SimpleImputer(strategy=imputation)), ("variance_filter", VarianceThreshold(threshold=variance_threshold))]
    if scaling:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


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
