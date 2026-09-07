import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_form as form
import generate_experiment as generator


def request(handler, method, path, values=None, token=None):
    instance = object.__new__(handler)
    instance.server = SimpleNamespace(server_port=8765)
    instance.path = path
    body = json.dumps(values).encode()
    instance.headers = {'Host': '127.0.0.1:8765', 'Content-Length': str(len(body))}
    if token:
        instance.headers['X-Form-Token'] = token
    instance.rfile = io.BytesIO(body)
    instance.wfile = io.BytesIO()
    statuses = []
    instance.send_response = statuses.append
    instance.send_header = lambda *args: None
    instance.end_headers = lambda: None
    getattr(instance, 'do_' + method)()
    return statuses[0], instance.wfile.getvalue()


def test_form_generates_notebook_and_rejects_invalid_choices(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, 'PROJECT_ROOT', tmp_path)
    handler = form.make_handler(['Erlotinib'], ['lung_NSCLC'], root=tmp_path)
    status, body = request(handler, 'GET', '/options')
    assert status == 200
    options = json.loads(body)
    values = dict(drug='Erlotinib', tissue='lung_NSCLC', metric='LN_IC50', min_cell_lines=20, notes='My hypothesis')
    assert request(handler, 'POST', '/generate', values)[0] == 403
    status, body = request(handler, 'POST', '/generate', values, options['token'])
    assert status == 201
    path = tmp_path / json.loads(body)['path']
    notebook = json.loads(path.read_text())
    assert notebook['metadata']['gdsc_experiment']['response_metric'] == 'LN_IC50'
    assert 'My hypothesis' in notebook['metadata']['gdsc_experiment']['prompt']
    for key, value in [('drug', 'typo'), ('tissue', 'typo'), ('metric', 'bad'), ('min_cell_lines', 2), ('min_cell_lines', 20.5)]:
        assert request(handler, 'POST', '/generate', {**values, key: value}, options['token'])[0] == 400
    assert len(list(path.parent.glob('*.ipynb'))) == 1


def test_form_page_and_unknown_route():
    handler = form.make_handler(['Example'], ['pancreas'])
    status, body = request(handler, 'GET', '/')
    assert status == 200
    assert b'<select id="drug"' in body
    assert b"\\nOpen this notebook" in body
    assert request(handler, 'GET', '/missing')[0] == 404
