# GDSC drug-response project

This project studies prediction of single-agent anticancer drug response from
genomic features, with performance compared across cancer/tissue types.

## Analysis notebooks

Read the notebooks in this order:

1. [`notebooks/01_main_analysis.ipynb`](notebooks/01_main_analysis.ipynb) is
   the end-to-end primary computational experiment: ingestion, preprocessing,
   model development, locked held-out evaluation, and feature interpretation.
2. [`notebooks/02_biological_context.ipynb`](notebooks/02_biological_context.ipynb)
   is the separate, targeted literature-review framework for contextualizing
   locked model features. It loads saved interpretation outputs and does not
   rerun or alter the computational experiment.
3. [`notebooks/03_sensitivity_ln_ic50.ipynb`](notebooks/03_sensitivity_ln_ic50.ipynb)
   is an unexecuted scaffold for the future LN_IC50 response-metric sensitivity
   analysis. The AUC experiment remains frozen.

The former long-form notebook, `notebook/gdsc_drug_response.ipynb`, is retained
unchanged as a historical reference while the split notebooks are adopted.

The biological-context notebook reads two small, versioned derived results:
`data/processed/locked_ridge_feature_interpretation_top20.csv` and
`data/processed/locked_ridge_top20_correlations.csv`. They preserve the locked
top-20 displays without rerunning model selection or the held-out evaluation.

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

### Initial NSCLC coverage analysis

The first cohort is `lung_NSCLC`, selected because it has the largest observed
number of GDSC cell lines (108). This is a documented initial analysis choice,
not a hard-coded restriction. `summarize_tissues`, `summarize_drugs`,
`response_duplicate_diagnostics`, and `drug_coverage_distribution` describe
coverage without selecting a drug or imposing an eligibility threshold.
`N_CELL_LINES` is the primary drug-level sample-size statistic; response rows
can repeat a drug/cell-line pair and are diagnosed, not averaged or discarded.

### Approved initial preprocessing experiment

`filter_eligible_drugs(drug_summary, min_unique_cell_lines=...)` implements a
configurable eligibility rule based on **unique cell lines**, never raw response
rows. The approved initial threshold is 75. `select_initial_drug` then selects
the eligible compound with the greatest cell-line coverage, breaking an exact
tie by its lowest `DRUG_ID`; this is a reproducible availability rule, not a
claim of biological superiority.

For this experiment, AUC is the target (with `LN_IC50` retained for later
sensitivity analysis). `select_response_dataset` chooses the single GDSC
screen with the greatest number of usable cell lines for the chosen drug; GDSC1
wins only an exact tie. Measurements from GDSC1 and GDSC2 are not averaged. A
remaining duplicate `DRUG_ID × COSMIC_ID` pair *within* the chosen screen is a
hard error, so the final response cohort always has one response per cell line.

### Verified preprocessing handoff

The approved initial experiment resolves to **Erlotinib** from **GDSC2** using
**AUC**: the GDSC2-specific stable identifier is `DRUG_ID=1168`. Its response
cohort contains 108 unique lung_NSCLC cell lines, with no missing AUC values or
within-screen duplicate drug/cell-line pairs. Targeted COSMIC mapping retains
106 cell lines with expression. The real, target-independent filtered matrix
has 16,980 expression features. The reproducible grouped split (`random_state`
42; 20% validation and 20% test) produces 63 training, 21 validation, and 22
test cell lines.

The training partition alone fits median imputation and the variance filter;
validation and test are transformed using those learned training statistics.
Scaling is disabled because COSMIC expression values are already Z-scores. The
notebook creates `X_train`, `X_val`, `X_test`, their aligned targets and
metadata, and the fitted preprocessing pipeline. It intentionally stops before
any predictive model is fit.

## Baseline modeling

The baseline stage compares three fixed, validation-only models: a mean-response
`DummyRegressor`, Ridge (`alpha=1.0`), and Elastic Net (`alpha=1.0`,
`l1_ratio=0.5`). Model construction lives in `gdsc.models`; MAE, RMSE, Pearson,
Spearman, and R² evaluation lives in `gdsc.evaluation`. Undefined correlations
from constant predictions are explicitly reported as `NaN`.

For the first experiment, the 63 training cell lines and 16,980 features give a
feature-to-sample ratio of about 270:1, motivating regularized linear models.
The 21-cell-line validation comparison found Ridge improved on the mean
baseline (MAE 0.066239 vs. 0.068166; RMSE 0.081292 vs. 0.088695; Pearson
0.445491). Fixed Elastic Net matched the mean baseline in this initial run.
These are development diagnostics, not biological conclusions or final model
selection. The 22-cell-line test partition remains untouched.

### Final locked-model held-out evaluation

Before opening the test set, the strategy was locked as Ridge with
`alpha=100.0`: it had the best validation RMSE (0.081280) and the lowest
three-fold training-CV RMSE variability (0.102637 ± 0.010035). The final rule
was to fit the already configured model on the 63-cell-line training partition
only, using the already-fitted training-only median imputer and variance filter.
No hyperparameter search, feature change, preprocessing refit, or model-family
change is permitted after this point.

The one-time held-out evaluation on 22 cell lines yielded MAE/RMSE of
0.050805/0.060515 for locked Ridge, versus 0.054069/0.061048 for the
training-mean baseline. Ridge Pearson and Spearman correlations were 0.354313
and 0.247883 (the constant baseline correlations are undefined and reported as
`NaN`). These are the primary generalization results for this initial
experiment; they represent a modest error improvement, not biological evidence.

## COSMIC expression feature store

COSMIC v104 Cell Lines Project expression is cached separately as long-format
`data/processed/cosmic_expression.parquet`. It is never merged onto the full
GDSC response table. Configure either a user-specific signed `COSMIC_LINK` or
`COSMIC_AUTHORIZATION` locally in `.env` (never commit either), then build the
cache once with `build_expression_cache("data")`.
Use `load_expression_features("data", cosmic_sample_ids=[...], genes=[...])`
to retrieve only the features required by a later preprocessing step.

### File roles: source text, temporary SQLite, and Parquet

`data/raw/` holds the downloaded, source-format files: GDSC response CSVs,
the GDSC metadata workbook, and COSMIC compressed TSV files. It may also
contain the pre-existing legacy wide COSMIC Parquet matrix generated during
earlier project work. The loader recognizes that file in its actual raw-data
location and reads only requested samples/genes, converting the bounded result
to the common long-format API. New cache builds write the preferred long-format
Parquet store under `data/processed/`. While `build_expression_cache` reads
the large COSMIC TSV in chunks, it uses a temporary SQLite database in the
system temporary directory solely to accumulate duplicate sample/gene Z-scores
without holding the full source in memory. This transient database is deleted
after the cache build; it is neither an analytical dataset nor a project
artifact. The durable output is the de-duplicated *long-format* Parquet file in
`data/processed/`. Parquet supports efficient predicate queries for the selected
COSMIC sample IDs (and, optionally, genes). Only after the response cohort is
fixed does preprocessing pivot that bounded query into a cell-line-by-gene
matrix; it never writes or creates a full response-row-by-gene table.

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

## Feature interpretation

The locked Ridge model (`alpha=100.0`) is interpreted by absolute signed
coefficients on the unchanged COSMIC Z-score feature set. Coefficients indicate
conditional predictive association with predicted AUC, not biological causation
or a sensitivity/resistance mechanism. Training-only bootstrap refits report
coefficient variation and sign consistency without tuning, changing features,
or using test responses.

Top-20 feature correlations are computed from training expression only and
reported numerically without removing any genes. Correlation matters because
Ridge can distribute predictive weight across correlated expression features;
therefore a large individual coefficient need not mean that one gene alone is
responsible for a prediction.
