# GDSC drug-response project

This project studies prediction of single-agent anticancer drug response from
genomic features, with performance compared across cancer/tissue types.

## Data source

The data layer uses the official Wellcome Sanger Institute CancerRxGene FTP
release **GDSC 8.4**, dated 24 July 2022:

`https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/`

It downloads the complete defined single-agent release components:

- `GDSC1_fitted_dose_response_24Jul22.csv`
- `GDSC2_fitted_dose_response_24Jul22.csv`
- `Cell_Lines_Details.xlsx`

The response files provide `AUC` and `LN_IC50`, along with drug, cell-line,
and COSMIC identifiers. The metadata workbook provides tissue descriptors and
TCGA-matched cancer type. Raw files are preserved under `data/raw/` and are
ignored by Git. A manifest records the release and source files.

The release workbook records availability flags for WES, CNA, gene expression,
and methylation. The project has not yet selected a genomic feature matrix;
that matching decision is intentionally deferred to the research-design stage.

## Data API

```python
from gdsc.data import download_gdsc, filter_by_tissue, load_gdsc, validate_gdsc

download_gdsc("../data/raw")
validate_gdsc("../data/raw")
gdsc = load_gdsc("../data/raw")
pancreas = filter_by_tissue(gdsc, "pancreas")
```

`download_gdsc` downloads the complete release and skips files already present.
`load_gdsc` combines GDSC1 and GDSC2 and joins cell-line metadata by
`COSMIC_ID`. Tissue is metadata, not a download parameter. `validate_gdsc`
performs offline structural checks. `prepare_gdsc` is a convenience wrapper
for download, validation, and loading.


## Preprocessing

The checked-in raw data contain response records and cell-line assay
availability flags, but no genomic feature matrix. A genomic source and
modality (WES, CNA, expression, or methylation) must be selected and supplied
before an analytical dataset can be created; the preprocessing module does not
silently substitute one.

`X` contains numeric genomic features only. `y` is the requested response
metric and `metadata` separately preserves identifiers, tissue, cancer type,
and drug fields. The returned `info` dictionary records join counts, feature
filtering, and thresholds. By default features with more than 20% missingness
and constant features are removed; remaining missing values are preserved.
This avoids a global imputation fit. If later modelling requires imputation or
scaling, call `build_training_transformer(...)`, fit it on training features
only, and use that fitted transformer to transform held-out data.

### Drug-specific preprocessing and leakage control

The modelling observation unit is **one cell line for one selected drug**.
`build_drug_dataset` first selects a tissue, applies explicit per-drug
observation/cell-line eligibility thresholds, maps only the remaining cell
lines to COSMIC, and requests only specified genes. It returns expression-only
`X`, the explicitly selected `AUC` or `LN_IC50` target `y`, and separate
metadata. Missingness and variance helpers expose filtering choices rather
than silently dropping values.

`split_by_cell_line` uses grouped splitting on `COSMIC_ID`, so a cell line can
never occur in more than one train/validation/test split. `build_preprocessor`
returns an unfitted imputer/variance-filter/optional-scaler pipeline; fit it on
the training split only before transforming validation or test data.

## COSMIC expression feature store

COSMIC v104 Cell Lines Project expression is cached separately as long-format
`data/processed/cosmic_expression.parquet`. It is never merged onto the full
GDSC response table. Configure either a user-specific signed `COSMIC_LINK` or
`COSMIC_AUTHORIZATION` locally in `.env` (never commit either), then build the
cache once with `build_expression_cache("data")`.
Use `load_expression_features("data", cosmic_sample_ids=[...], genes=[...])`
to retrieve only the features required by a later preprocessing step.

### Why expression is queried rather than globally joined

An earlier ingestion attempt merged the complete COSMIC expression matrix
(about 17,000 genes) onto every GDSC drug-response observation (about 575,000
rows). This would create a dense response-row-by-gene table and failed with a
memory allocation request of roughly 72.8 GiB.

The replacement is a feature-store design: GDSC responses and metadata remain
lightweight; COSMIC expression is de-duplicated in long format using the
documented arithmetic mean for repeated sample/gene Z-scores and cached once in
`data/processed/cosmic_expression.parquet`. Later preprocessing first selects
the relevant cohort, cell lines, and genes, then queries only those rows using
`load_expression_features`. The final modelling-table merge is therefore small,
explicit, and performed only after feature restrictions have been chosen.
