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
        if cell["cell_type"] == "code" and "runtime-setup" not in cell["metadata"].get("tags", []):
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
        if cell["cell_type"] == "code" and "runtime-setup" not in cell["metadata"].get("tags", []):
            exec(compile(cell["source"], "generated", "exec"), namespace)
    assert len(namespace["held_out_results"]) == 2
    assert namespace["predictions"].response_metric.eq(metric).all()
    assert "constant" not in namespace["feature_ranking"].GENE_SYMBOL.tolist()
    assert len(namespace["feature_ranking"]) == 5


def test_runtime_setup_supports_ipython_magics():
    from IPython.core.interactiveshell import InteractiveShell
    cells = generator.notebook_setup_cells()
    source = cells[1]["source"]
    assert "%pip install -e" in source
    assert "src" in source and "git" in source
    transformed = InteractiveShell.instance().transform_cell(source)
    compile(transformed, "runtime-setup", "exec")


@pytest.mark.parametrize("location", ["project", "notebook"])
def test_run_all_loads_dotenv_before_cosmic_request(tmp_path, monkeypatch, capsys, location):
    import sys
    from gdsc import cosmic

    project = tmp_path / "project"
    notebook = project / "notebooks"
    notebook.mkdir(parents=True)
    env_dir = project if location == "project" else notebook
    (env_dir / ".env").write_text('\ufeffCOSMIC_AUTHORIZATION=fixture-run-all-token\n')
    monkeypatch.setenv("COSMIC_AUTHORIZATION", "")
    monkeypatch.setenv("COSMIC_LINK", "")
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    cell = next(c for c in generator.notebook_setup_cells() if c["id"] == "runtime-env")
    exec(cell["source"], {"setup_root": project, "setup_env_dir": notebook})
    assert cosmic._cosmic_request().get_header("Authorization") == "Basic fixture-run-all-token"
    assert "fixture-run-all-token" not in capsys.readouterr().out


def test_checked_in_notebooks_load_credentials_before_analysis():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "notebook/gdsc_drug_response.ipynb", *sorted((root / "notebooks").rglob("*.ipynb"))]
    for path in paths:
        cells = json.loads(path.read_text())["cells"]
        expected = generator.notebook_setup_cells(str(path.parent.relative_to(root)))
        for actual, template in zip(cells[:4], expected):
            assert actual["id"] == template["id"], path
            assert "".join(actual["source"]) == template["source"], path


def test_colab_env_upload_loads_values_and_removes_file(monkeypatch, capsys, tmp_path):
    import os
    import types
    import sys
    uploads = []

    def upload_file(filename):
        path = Path(filename)
        uploads.append(path)
        path.write_text('GDSC_TEST_UPLOAD=example-private-value\nCOSMIC_AUTHORIZATION=example-test-token\n')

    colab = types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(upload_file=upload_file)
    class MissingSecret(Exception):
        pass
    def missing(key):
        raise MissingSecret(key)
    colab.userdata = types.SimpleNamespace(get=missing, SecretNotFoundError=MissingSecret, NotebookAccessError=MissingSecret)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setenv('GDSC_TEST_UPLOAD', 'previous')
    monkeypatch.setenv('COSMIC_AUTHORIZATION', '')
    monkeypatch.delenv('COSMIC_LINK', raising=False)
    cell = next(c for c in generator.notebook_setup_cells() if c['id'] == 'runtime-env')
    exec(cell['source'], {'setup_root': tmp_path})
    assert os.environ['GDSC_TEST_UPLOAD'] == 'example-private-value'
    assert not uploads[0].exists()
    output = capsys.readouterr().out
    assert 'example-private-value' not in output
    assert 'example-test-token' not in output
    assert 'COSMIC_AUTHORIZATION: set' in output


def test_run_all_reuses_colab_secrets_without_upload(monkeypatch, tmp_path, capsys):
    import os
    import sys
    import types
    monkeypatch.delenv("COSMIC_AUTHORIZATION", raising=False)
    monkeypatch.delenv("COSMIC_LINK", raising=False)
    calls = []
    def get(key):
        calls.append(key)
        return "test-secret-value"
    def upload(*args):
        pytest.fail("Run all should not prompt when a secret is available")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload_file=upload)
    colab.userdata = types.SimpleNamespace(get=get, SecretNotFoundError=KeyError, NotebookAccessError=PermissionError)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    cell = next(c for c in generator.notebook_setup_cells() if c["id"] == "runtime-env")
    namespace = {"setup_root": tmp_path}
    exec(cell["source"], namespace)
    exec(cell["source"], namespace)
    assert calls == ["COSMIC_AUTHORIZATION"]
    assert os.environ["COSMIC_AUTHORIZATION"] == "test-secret-value"
    assert "test-secret-value" not in capsys.readouterr().out
