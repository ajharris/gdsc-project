import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "generate_experiment", Path(__file__).resolve().parents[1] / "scripts/generate_experiment.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_prompt_matching_and_ambiguity():
    assert generator.resolve_name("Study ERLOTINIB in lung NSCLC", ["Erlotinib"], "drug") == "Erlotinib"
    assert generator.resolve_name("Study lung NSCLC", ["lung", "lung_NSCLC"], "tissue") == "lung_NSCLC"
    with pytest.raises(ValueError, match="Specify one drug"):
        generator.resolve_name("A and B", ["A", "B"], "drug")
    with pytest.raises(ValueError, match="No matching"):
        generator.resolve_name("unknown", ["Erlotinib"], "drug")
    assert generator.resolve_name("A and B", ["A", "B"], "drug", "b") == "B"


def test_cli_creates_unique_clean_notebooks(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generator, "catalog", lambda _: (["Erlotinib"], ["lung_NSCLC"]))
    for _ in range(2):
        assert generator.main(["Study Erlotinib in lung NSCLC"]) == 0
    paths = sorted((tmp_path / "notebooks/experiments").glob("*.ipynb"))
    assert len(paths) == 2
    notebook = json.loads(paths[0].read_text())
    assert notebook["metadata"]["gdsc_experiment"]["drug"] == "Erlotinib"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile(cell["source"], "generated", "exec")


@pytest.mark.parametrize("metric", ["AUC", "LN_IC50"])
def test_generated_notebook_executes_with_synthetic_cohort(monkeypatch, metric):
    from gdsc import data, preprocessing
    from gdsc.preprocessing import AnalysisDataset

    rng = np.random.default_rng(42)
    count = 40
    response = pd.DataFrame({
        "COSMIC_ID": range(count), "DRUG_ID": 1, "DRUG_NAME": "Example",
        "DATASET": "GDSC2", "TISSUE_OF_ORIGIN": "lung_NSCLC",
        "AUC": rng.normal(size=count), "LN_IC50": rng.normal(size=count),
    })
    X = pd.DataFrame(rng.normal(size=(count, 5)), columns=list("ABCDE"))
    X["constant"] = 1.0
    X.loc[0, "A"] = np.nan
    monkeypatch.setattr(data, "prepare_gdsc", lambda _: response)
    monkeypatch.setattr(preprocessing, "build_expression_dataset", lambda cohort, **kwargs:
                        AnalysisDataset(X, cohort[metric], cohort, {}))
    namespace = {}
    notebook = generator.make_notebook("Example in lung NSCLC", "Example", "lung_NSCLC", metric, 20)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(cell["source"], "generated", "exec"), namespace)
    assert len(namespace["held_out_results"]) == 2
    assert namespace["predictions"].response_metric.eq(metric).all()
    assert "constant" not in namespace["feature_ranking"].GENE_SYMBOL.tolist()
    assert len(namespace["feature_ranking"]) == 5
