#!/usr/bin/env python3
"""Serve a local form for choosing and generating a GDSC experiment notebook."""
from __future__ import annotations

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer

from generate_experiment import PROJECT_ROOT, catalog, save_experiment

PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDSC experiment builder</title>
<style>
body{font:16px system-ui,sans-serif;background:#f3f6fa;color:#172b42;margin:0;padding:32px 16px}
main{max-width:720px;margin:auto;background:white;padding:32px;border-radius:16px}
h1{margin-top:0}p{line-height:1.6;color:#42566c}label{display:block;font-weight:600;margin-top:22px}
input,select,textarea,button{box-sizing:border-box;width:100%;font:inherit;padding:12px;border:1px solid #8294a8;border-radius:6px;margin-top:8px}
select[size]{height:170px}button{background:#174f91;color:white;cursor:pointer;border:0;margin-top:24px}
button:disabled{opacity:.5;cursor:wait}#status{white-space:pre-wrap;overflow-wrap:anywhere;padding-top:20px}
small{display:block;color:#42566c;margin-top:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:550px){.grid{grid-template-columns:1fr}main{padding:20px}}
</style>
<main><h1>Build an experiment</h1><p>Choose a tissue and drug from your GDSC data.
Generate a notebook, then open it in your editor to run the experiment.</p>
<form id="form">
<label for="tissueSearch">Find a tissue</label><input id="tissueSearch" type="search" placeholder="Filter tissues…">
<label for="tissue">Tissue</label><select id="tissue" size="5" required></select>
<label for="drugSearch">Find a drug</label><input id="drugSearch" type="search" placeholder="Filter drugs…">
<label for="drug">Drug</label><select id="drug" size="5" required></select>
<small>These lists show all cached names. The notebook checks coverage for your selected pair.</small>
<div class="grid"><div><label for="metric">Response metric</label><select id="metric"><option>AUC</option><option>LN_IC50</option></select></div>
<div><label for="minimum">Minimum cell lines</label><input id="minimum" type="number" min="10" step="1" value="20" required></div></div>
<label for="notes">Research notes (optional)</label><textarea id="notes" rows="3" maxlength="10000" placeholder="What do you want to investigate?"></textarea>
<small>Notes are recorded in the notebook; they do not change its analysis steps.</small>
<button id="submit" disabled>Create notebook</button></form><div id="status" role="status" aria-live="polite">Loading choices…</div></main>
<script>
const $ = id => document.getElementById(id);
let token;
function choices(id, names) {
  const select = $(id), search = $(id+'Search');
  function refresh() {
    const previous = select.value, query = search.value.toLowerCase().replaceAll('_',' ');
    select.replaceChildren();
    names.filter(name => name.toLowerCase().replaceAll('_',' ').includes(query)).forEach(name => {
      const option = new Option(name, name); option.selected = name === previous; select.add(option);
    });
    if (!previous || !Array.from(select.options).some(o => o.value === previous)) select.selectedIndex = -1;
  }
  search.addEventListener('input', refresh); refresh();
}
fetch('/options').then(r => r.json()).then(data => {
  token = data.token; choices('drug', data.drugs); choices('tissue', data.tissues);
  $('status').textContent = ''; $('submit').disabled = false;
}).catch(() => $('status').textContent = 'Could not load choices. Restart the form and reload this page.');
$('form').addEventListener('submit', async event => {
  event.preventDefault(); $('submit').disabled = true; $('status').textContent = 'Creating notebook…';
  try {
    const response = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json','X-Form-Token':token},
      body:JSON.stringify({drug:$('drug').value,tissue:$('tissue').value,metric:$('metric').value,
        min_cell_lines:Number($('minimum').value),notes:$('notes').value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    $('status').textContent = 'Created '+result.path+'\nOpen this notebook in your editor and run its cells in order.';
  } catch(error) { $('status').textContent = error.message; }
  finally { $('submit').disabled = false; }
});
</script></html>"""


def make_handler(drugs, tissues, *, root=PROJECT_ROOT):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, *, html=False):
            body = value.encode() if html else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {"error": "Open the form using its printed local address."})
            elif self.path == "/":
                self.reply(200, PAGE, html=True)
            elif self.path == "/options":
                self.reply(200, {"drugs": drugs, "tissues": tissues, "token": token})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host() or not secrets.compare_digest(self.headers.get("X-Form-Token", ""), token):
                self.reply(403, {"error": "Reload the local form before submitting."})
                return
            if self.path != "/generate":
                self.reply(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 64_000:
                    raise ValueError("Form submission is too large or empty")
                values = json.loads(self.rfile.read(size))
                if not isinstance(values, dict):
                    raise ValueError("Invalid form submission")
                drug, tissue = values.get("drug"), values.get("tissue")
                if drug not in drugs or tissue not in tissues:
                    raise ValueError("Choose a drug and tissue from the lists")
                minimum = values.get("min_cell_lines")
                if type(minimum) is not int or minimum < 10:
                    raise ValueError("Minimum cell lines must be an integer of at least 10")
                metric = values.get("metric")
                if metric not in ("AUC", "LN_IC50"):
                    raise ValueError("Choose AUC or LN_IC50")
                notes = values.get("notes", "")
                if not isinstance(notes, str) or len(notes) > 10_000:
                    raise ValueError("Notes must be text of at most 10,000 characters")
                prompt = f"Predict {drug} response in {tissue}" + (f"\n\n{notes.strip()}" if notes.strip() else "")
                path = save_experiment(prompt, drug, tissue, metric, minimum)
                self.reply(201, {"path": str(path.relative_to(root))})
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except OSError:
                self.reply(500, {"error": "Could not write the notebook. Check folder permissions and disk space."})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    try:
        drugs, tissues = catalog(PROJECT_ROOT / "data/raw")
        server = HTTPServer(("127.0.0.1", args.port), make_handler(drugs, tissues))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Open http://127.0.0.1:{server.server_port} in your browser. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
