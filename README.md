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
