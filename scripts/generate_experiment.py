#!/usr/bin/env python3
"""Generate an unexecuted GDSC experiment from a drug-and-tissue prompt."""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def normalise(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def resolve_name(prompt: str, names: list[str], kind: str, explicit: str | None = None) -> str:
    """Match whole names, allowing case, spaces, underscores and hyphens."""
    query = normalise(explicit if explicit is not None else prompt)
    matches = []
    for name in names:
        label = normalise(name)
        if label and (query == label if explicit is not None else f" {label} " in f" {query} "):
            matches.append(name)
    # Prefer a full compound name over another name contained inside it.
    if explicit is None:
        matches = [name for name in matches if not any(
            normalise(name) != normalise(other)
            and f" {normalise(name)} " in f" {normalise(other)} "
            for other in matches
        )]
    if len(matches) != 1:
        detail = f"Matches: {sorted(matches)}" if matches else "No matching GDSC name found."
        raise ValueError(f"Specify one {kind} using --{kind}. {detail} Use --list-options to see names.")
    return matches[0]


def catalog(raw_dir: Path) -> tuple[list[str], list[str]]:
    import pandas as pd
    from gdsc.data import SOURCE_FILES, _load_cell_line_metadata

    paths = sorted(raw_dir.glob("GDSC[12]_fitted_dose_response_*.csv"))
    metadata = raw_dir / SOURCE_FILES["cell_lines"]
    if not paths or not metadata.exists():
        raise FileNotFoundError(f"Cached GDSC response CSVs and metadata are required in {raw_dir}. Run download_gdsc first.")
    drugs = set()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["DRUG_NAME"], chunksize=100_000):
            drugs.update(chunk.DRUG_NAME.dropna().unique())
    tissues = _load_cell_line_metadata(metadata).TISSUE_OF_ORIGIN.dropna().unique()
    return sorted(drugs), sorted(tissues)


def notebook_setup_cells(notebook_directory="notebooks/experiments") -> list[dict]:
    """Bootstrap a repository checkout and install its package in a notebook kernel."""
    explanation = (
        "## Runtime setup (Colab or local)\n\n"
        "Run this cell before the analysis. It reuses a local checkout or clones the "
        "GitHub repository into Colab, then installs `src/gdsc` and the dependencies "
        "declared in `pyproject.toml`. Python 3.12 or later is required.\n\n"
        "The clone contains the published default branch; push your changes before "
        "opening a fresh Colab runtime. For a private repository, provide an authenticated "
        "checkout at `/content/gdsc-project` first. Raw data and local `.env` credentials "
        "are not included. Configure COSMIC access in the runtime before downloading "
        "expression data. If packages were already imported, restart the runtime after "
        "installation and rerun this cell.\n"
    )
    source = textwrap.dedent("""\
        import sys
        import subprocess
        from pathlib import Path

        setup_env_dir = Path.cwd()
        if sys.version_info < (3, 12):
            raise RuntimeError("This project requires Python 3.12 or later; select a compatible runtime.")
        setup_root = next((p for p in (Path.cwd(), *Path.cwd().parents)
                           if (p / "src/gdsc/data.py").is_file() and (p / "pyproject.toml").is_file()), None)
        if setup_root is None:
            if "google.colab" not in sys.modules:
                raise RuntimeError("Open this notebook from within the project checkout.")
            setup_root = Path("/content/gdsc-project")
            if not setup_root.exists():
                subprocess.run(["git", "clone", "https://github.com/ajharris/gdsc-project.git", str(setup_root)], check=True)
            if not (setup_root / "src/gdsc/data.py").is_file():
                raise RuntimeError("Expected a project checkout at /content/gdsc-project.")

        # Editable installation includes src/gdsc and all declared runtime dependencies.
        %pip install -e "$setup_root"
        sys.path.insert(0, str(setup_root / "src"))
    """)
    source += f"setup_notebook_dir = setup_root / {notebook_directory!r}\n"
    source += "setup_notebook_dir.mkdir(parents=True, exist_ok=True)\n%cd $setup_notebook_dir\n"
    env_source = textwrap.dedent("""\
        # Run before analysis; rerunning also refreshes imported COSMIC configuration.
        import sys
        import os
        import importlib
        import tempfile
        from pathlib import Path
        from dotenv import load_dotenv

        def load_runtime_environment():
            def credentials_present():
                return any(os.environ.get(key, "").strip()
                           for key in ("COSMIC_AUTHORIZATION", "COSMIC_LINK"))

            # Local config takes precedence over defaults; existing runtime values
            # and accessible Colab Secrets also work without an upload prompt.
            env_candidates = [setup_root / ".env",
                              globals().get("setup_env_dir", setup_root) / ".env"]
            if "google.colab" in sys.modules:
                env_candidates.append(Path("/content/.env"))
            for env_path in dict.fromkeys(env_candidates):
                if env_path.is_file():
                    load_dotenv(env_path, override=True, encoding="utf-8-sig")
                    if credentials_present():
                        break
            if "google.colab" in sys.modules:
                from google.colab import files, userdata
                if not credentials_present():
                    for key in ("COSMIC_AUTHORIZATION", "COSMIC_LINK"):
                        try:
                            value = userdata.get(key)
                        except userdata.TimeoutException:
                            print("Colab Secrets unavailable. Place .env in /content/.env when running outside the Colab UI.")
                            break
                        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                            continue
                        if value and value.strip():
                            os.environ[key] = value.strip()
                            break
                if not credentials_present():
                    print("No credentials available. Select your .env once to continue Run all.")
                    with tempfile.TemporaryDirectory(prefix="gdsc-env-") as directory:
                        env_path = Path(directory) / ".env"
                        files.upload_file(str(env_path))
                        load_dotenv(env_path, override=True, encoding="utf-8-sig")
            load_dotenv(setup_root / ".env.example", encoding="utf-8-sig")
            authorization_present = bool(os.environ.get("COSMIC_AUTHORIZATION", "").strip())
            link_present = bool(os.environ.get("COSMIC_LINK", "").strip())
            print("COSMIC_AUTHORIZATION:", "set" if authorization_present else "missing or empty")
            print("COSMIC_LINK:", "set" if link_present else "missing or empty")
            if not (authorization_present or link_present):
                raise RuntimeError("No COSMIC credentials loaded. Upload your actual .env, not .env.example; it must contain COSMIC_AUTHORIZATION or COSMIC_LINK.")
            # COSMIC keeps file configuration in module constants. Refresh it if this
            # cell was run after an earlier analysis import in the same runtime.
            if "gdsc.cosmic" in sys.modules:
                importlib.reload(sys.modules["gdsc.cosmic"])
                print("Refreshed imported COSMIC configuration. Rerun subsequent analysis cells.")
            print("Environment loaded. Variable values are not displayed.")

        load_runtime_environment()
    """)
    return [
        {"cell_type": "markdown", "id": "runtime-setup-intro", "metadata": {}, "source": explanation},
        {"cell_type": "code", "id": "runtime-setup", "metadata": {"tags": ["runtime-setup"]},
         "source": source, "execution_count": None, "outputs": []},
        {"cell_type": "markdown", "id": "runtime-env-intro", "metadata": {},
         "source": "### Run all: configure credentials once\n\n"
                   "For unattended Colab setup, open **Secrets** (the key icon), add `COSMIC_AUTHORIZATION` "
                   "with the value from your local `.env`, and enable **Notebook access**. "
                   "Then choose **Runtime → Run all**. A signed `COSMIC_LINK` secret is also supported, "
                   "but expires and may need replacement.\n\n"
                   "For `.env` setup, place your file in the project root or beside the notebook before Run all. "
                   "In Colab, upload it to `/content/.env` using the Files panel before Run all. "
                   "Setup reuses that `.env` or runtime credentials, then checks Colab Secrets. "
                   "If neither is available, Run all pauses at a file picker for your `.env` and continues after upload. "
                   "Local `.env` files cannot be read automatically from a remote Colab server. "
                   "Files selected through the setup picker are removed after loading; secrets are never printed. "
                   "Saved Colab Secrets can be used again after a runtime restart without another upload.\n"},
        {"cell_type": "code", "id": "runtime-env", "metadata": {"tags": ["runtime-setup"]},
         "source": env_source, "execution_count": None, "outputs": []},
    ]


def make_notebook(prompt: str, drug: str, tissue: str, metric: str, min_cell_lines: int) -> dict:
    cells = notebook_setup_cells()

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"# {drug} response in {tissue}\n\n"
        f"Predict {metric} from COSMIC expression for this drug/tissue cohort.\n\n"
        "Original prompt (recorded as text; additional instructions do not change the template):\n\n"
        + "\n".join("> " + line for line in prompt.splitlines()))
    add("markdown", """
        ## Experiment configuration
        GDSC 8.4 responses and COSMIC v104 expression are reused from the project data
        directory. Running the notebook downloads missing sources; COSMIC requires local
        credentials in `.env`. Generation itself does not download or run the experiment.
        AUC is the default target; configure alternatives before examining results.
        The minimum cohort size is an execution guard, not a statistical power claim.
    """)
    add("code", f"DRUG = {drug!r}\nTISSUE = {tissue!r}\nRESPONSE_METRIC = {metric!r}\n"
        f"MIN_CELL_LINES = {min_cell_lines}\nRANDOM_STATE = 42\nRIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]")
    add("code", """
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        from IPython.display import display

        project_root = next((path for path in (Path.cwd(), *Path.cwd().parents)
                             if (path / "src/gdsc/data.py").is_file()), None)
        if project_root is None:
            raise RuntimeError("Launch the notebook from this repository or one of its subdirectories.")
        sys.path.insert(0, str(project_root / "src"))
        from gdsc.data import prepare_gdsc
        from gdsc import preprocessing
        from gdsc.evaluation import evaluate_regression
        from gdsc.models import build_dummy_regressor, build_ridge_model
        from gdsc.interpretation import ridge_coefficients

        data_dir = project_root / "data"
        gdsc = prepare_gdsc(data_dir / "raw")
    """)
    add("markdown", """
        ## Resolve the response cohort
        Use the requested drug, selecting the screen with the most usable cell lines
        (GDSC1 breaks a tie). Screens are not averaged. Within-screen duplicates are errors.
    """)
    add("code", """
        response = preprocessing.select_response_dataset(
            gdsc, tissue_of_origin=TISSUE, drug_name=DRUG, response_metric=RESPONSE_METRIC,
        )
        response_cohort = response["response_cohort"]
        display(response["dataset_coverage"])
        print({"screen": response["selected_dataset"], "drug_id": response["selected_drug_id"],
               "cell_lines": response_cohort.COSMIC_ID.nunique(),
               "excluded_missing_target": response["n_excluded_response_rows"]})
        if response_cohort.COSMIC_ID.nunique() < MIN_CELL_LINES:
            raise ValueError("Response cohort is below MIN_CELL_LINES; review coverage and study design.")
        display(response_cohort[RESPONSE_METRIC].describe())
    """)
    add("markdown", """
        ## Expression and grouped splits
        Query expression only for the resolved cohort. The existing helper applies
        target-independent missingness and constant-feature filtering across this cohort;
        this is a limitation relative to fitting all feature filters on training data.
        Median imputation and the subsequent variance filter are fitted on training only.
        Validation and test each receive approximately 20% of cell lines.
    """)
    add("code", """
        dataset = preprocessing.build_expression_dataset(
            response_cohort, response_metric=RESPONSE_METRIC, data_dir=data_dir,
            max_gene_missing_fraction=0.20,
        )
        print(dataset.diagnostics)
        if len(dataset.y) < MIN_CELL_LINES or dataset.X.shape[1] == 0:
            raise ValueError("Insufficient expression-matched cell lines or features; review diagnostics.")
        if not np.isfinite(dataset.y.to_numpy(dtype=float)).all():
            raise ValueError("Response contains non-finite values.")
        splits = preprocessing.split_by_cell_line(dataset, random_state=RANDOM_STATE)
        train, validation, test = (splits[name] for name in ("train", "validation", "test"))
        if min(len(part.y) for part in (train, validation, test)) < 2:
            raise ValueError("Each split needs at least two cell lines for evaluation.")
        split_ids = [set(part.metadata.COSMIC_ID) for part in (train, validation, test)]
        assert not any(split_ids[i] & split_ids[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        preprocessor = preprocessing.build_preprocessor(imputation="median", scaling=False)
        X_train = preprocessor.fit_transform(train.X)
        X_val = preprocessor.transform(validation.X)
        feature_names = preprocessor.get_feature_names_out(train.X.columns)
        display(pd.DataFrame({name: {"cell_lines": len(part.y), "input_genes": part.X.shape[1]}
                              for name, part in splits.items()}).T)
    """)
    add("markdown", """
        ## Validation-only development and model lock
        Compare the predefined Ridge grid and a training-mean baseline on validation RMSE.
        Choose Ridge alpha by lowest validation RMSE, breaking ties by smaller alpha.
        The mean baseline remains a comparator even if it outperforms Ridge.
        No test responses are used to choose the model.
    """)
    add("code", """
        baseline = build_dummy_regressor().fit(X_train, train.y)
        baseline_validation = evaluate_regression(validation.y, baseline.predict(X_val))
        fitted_ridges, validation_rows = {}, []
        for alpha in RIDGE_ALPHAS:
            model = build_ridge_model(alpha=alpha).fit(X_train, train.y)
            fitted_ridges[alpha] = model
            metrics = evaluate_regression(validation.y, model.predict(X_val))
            validation_rows.append({"alpha": alpha, **metrics.__dict__})
        validation_results = pd.DataFrame(validation_rows).sort_values(["rmse", "alpha"])
        display(validation_results)
        print({"mean_baseline_validation": baseline_validation.__dict__})
        locked_alpha = float(validation_results.iloc[0]["alpha"])
        locked_model = fitted_ridges[locked_alpha]
        print({"locked_model": "Ridge", "alpha": locked_alpha, "fit_partition": "train only"})
    """)
    add("markdown", """
        ## Held-out evaluation
        Run after development decisions are fixed. Evaluate the already fitted Ridge and
        mean baseline once. Do not use test results to revise this experiment; record any
        subsequent design as a separate experiment.
    """)
    add("code", """
        X_test = preprocessor.transform(test.X)
        test_rows = []
        predictions = test.metadata.copy()
        predictions["response_metric"] = RESPONSE_METRIC
        predictions["observed"] = test.y.to_numpy()
        for name, model in (("Mean baseline", baseline), ("Locked Ridge", locked_model)):
            predicted = model.predict(X_test)
            predictions[name] = predicted
            test_rows.append({"model": name, **evaluate_regression(test.y, predicted).__dict__})
        held_out_results = pd.DataFrame(test_rows).set_index("model")
        display(held_out_results)
        display(predictions)
    """)
    add("markdown", """
        ## Feature interpretation and conclusions
        Coefficients are predictive associations, not causal mechanisms. Correlated genes
        can share weights. Signs refer to the configured response metric. Small cohorts and
        many features limit generalization; cell-line results do not establish clinical benefit.
    """)
    add("code", """
        feature_ranking = ridge_coefficients(locked_model, feature_names)
        display(feature_ranking.head(20))
    """)
    add("markdown", """
        ## Experiment notes
        - Record response and expression coverage, exclusions, and split sizes.
        - Compare validation and held-out errors with the mean baseline.
        - Record whether the evidence supports useful prediction, including negative results.
        - Document limitations and biological hypotheses separately from observed results.
    """)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "gdsc_experiment": {"prompt": prompt, "drug": drug, "tissue": tissue,
                            "response_metric": metric, "template_version": 1},
    }, "nbformat": 4, "nbformat_minor": 5}


def save_experiment(prompt: str, drug: str, tissue: str, metric: str = "AUC", min_cell_lines: int = 20) -> Path:
    """Save a new notebook without overwriting an existing experiment."""
    if metric not in {"AUC", "LN_IC50"}:
        raise ValueError("Unsupported response metric")
    if min_cell_lines < 10:
        raise ValueError("Minimum cell lines must be at least 10")
    notebook = make_notebook(prompt, drug, tissue, metric, min_cell_lines)
    folder = PROJECT_ROOT / "notebooks/experiments"
    folder.mkdir(parents=True, exist_ok=True)
    stem = "_".join(normalise(value).replace(" ", "_") for value in (drug, tissue, metric))
    # Exclusive creation preserves previous experiments, including their outputs.
    for number in range(1, 100_000):
        path = folder / f"{stem}{'' if number == 1 else f'_{number}'}.ipynb"
        try:
            with path.open("x", encoding="utf-8") as output:
                json.dump(notebook, output, indent=2, ensure_ascii=False)
                output.write("\n")
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("Too many experiments with this name")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help='Example: "Predict Erlotinib response in lung NSCLC"')
    parser.add_argument("--drug", help="Explicit GDSC drug name (overrides prompt matching)")
    parser.add_argument("--tissue", help="Explicit GDSC tissue label (overrides prompt matching)")
    parser.add_argument("--metric", choices=["AUC", "LN_IC50"], default="AUC")
    parser.add_argument("--min-cell-lines", type=int, default=20)
    parser.add_argument("--list-options", action="store_true", help="List names from cached GDSC data and exit")
    args = parser.parse_args(argv)
    if args.min_cell_lines < 10:
        parser.error("--min-cell-lines must be at least 10 for the three-way split")
    try:
        drugs, tissues = catalog(PROJECT_ROOT / "data/raw")
        if args.list_options:
            print("Drugs:\n" + "\n".join(drugs) + "\n\nTissues:\n" + "\n".join(tissues))
            return 0
        prompt = args.prompt
        if prompt is None and args.drug and args.tissue:
            prompt = f"Predict {args.drug} response in {args.tissue}"
        if not prompt or not prompt.strip():
            parser.error("Provide a prompt, or both --drug and --tissue")
        drug = resolve_name(prompt, drugs, "drug", args.drug)
        tissue = resolve_name(prompt, tissues, "tissue", args.tissue)
        path = save_experiment(prompt, drug, tissue, args.metric, args.min_cell_lines)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Created {path}\nDrug: {drug} | Tissue: {tissue} | Metric: {args.metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
