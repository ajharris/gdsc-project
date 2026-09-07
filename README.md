# Predicting drug response with GDSC and COSMIC

Can gene expression help predict how cancer cell lines respond to a drug?
This project combines **GDSC drug-response measurements** with **COSMIC gene
expression**, builds predictive models for a selected drug and tissue, and
examines the genes associated with their predictions.

The first experiment studies **Erlotinib in non-small-cell lung cancer
(`lung_NSCLC`)**, using AUC as the response measure. It follows the analysis
from data preparation through evaluation on held-out cell lines. The broader
aim is to repeat this workflow across drugs and tissues. Gene associations
are starting points for biological investigation, not evidence of causation.

## Read the project from start to finish

1. **[Main analysis](notebooks/01_main_analysis.ipynb) — how the experiment works.**
   Start here for the research question, data sources, and cohort selection.
   Follow the notebook through matching cell lines to expression, separating
   training/validation/test data, comparing models, evaluating the final model,
   and interpreting its gene coefficients. This is the primary computational
   experiment and contains the recorded AUC results.

2. **[Biological context](notebooks/02_biological_context.ipynb) — what the findings might mean.**
   Read the saved feature rankings alongside a framework for reviewing evidence
   about Erlotinib, its pathways, and lung cancer. This notebook explains how to
   separate model findings from biological hypotheses. The literature review
   is a scaffold awaiting verified evidence; it does not retrain the model.

3. **[Response-metric sensitivity](notebooks/03_sensitivity_ln_ic50.ipynb) — what to investigate next.**
   This planned analysis asks whether conclusions change when drug response is
   measured with LN_IC50 instead of AUC. It is an unimplemented scaffold, with
   no sensitivity-analysis results yet.

The [original combined notebook](notebook/gdsc_drug_response.ipynb) is kept as
historical reference. Use the three notebooks above as the reading path.

## Run the notebooks

Use **Python 3.12 or later**. From the repository root, create and activate an
environment, then install the project and notebook kernel:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e . ipykernel
```

Open the main analysis notebook in your notebook editor, select this environment
as the kernel, and run the cells in order. It downloads missing GDSC 8.4 files
and builds or reuses the COSMIC v104 expression cache. For a fresh COSMIC download,
set `COSMIC_AUTHORIZATION` or a signed `COSMIC_LINK` in your local `.env`; see
[the configuration example](.env.example). Keep credentials out of version control.
The initial expression-cache build can take time; later runs reuse it.

## Create your own experiment

Once the GDSC files are cached, launch the local selection form from the activated
environment:

```bash
python scripts/experiment_form.py
```

Open **http://127.0.0.1:8765**, choose a drug and tissue from the searchable lists,
and click **Create notebook**. Open the new file in
[`notebooks/experiments/`](notebooks/experiments/) and run it with the project
kernel. The form generates an unexecuted analysis template; it does not run
models. Existing notebooks are preserved. Stop the form with Ctrl+C.

For command-line generation, use
[`generate_experiment.py`](scripts/generate_experiment.py):

```bash
python scripts/generate_experiment.py "Predict Erlotinib response in lung NSCLC"
```

Both tools support AUC or LN_IC50 experiments. Run either script with `--help`
for its options. New experiments have their own results and should be read
separately from the recorded primary analysis.

## Where the implementation lives

The notebooks explain the analysis; [`src/gdsc/`](src/gdsc/) contains its reusable
Python functions. Follow [data loading](src/gdsc/data.py) and
[expression caching](src/gdsc/cosmic.py) into
[preprocessing](src/gdsc/preprocessing.py), [models](src/gdsc/models.py),
[evaluation](src/gdsc/evaluation.py), and
[feature interpretation](src/gdsc/interpretation.py).
Raw downloads live in `data/raw/`; expression caches and derived results live
in `data/processed/`. Most data files are excluded from version control.
