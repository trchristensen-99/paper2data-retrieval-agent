from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.database.store import PaperDatabase
from src.utils.env import load_env_file


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Paper2Data Terminal</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #1b2635;
      --accent: #0a7ea4;
      --line: #d6dde6;
    }
    body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }
    .wrap { max-width: 1240px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 14px; font-size: 24px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .card .k { color: #5a6778; font-size: 12px; }
    .card .v { font-size: 21px; font-weight: 700; }
    .filters { display: grid; grid-template-columns: 2fr 1fr 1fr auto; gap: 8px; margin-bottom: 8px; }
    .filters.advanced { grid-template-columns: repeat(6, minmax(120px, 1fr)); display: none; }
    .filters.advanced.show { display: grid; }
    input, select, button { border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; }
    button { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eef2f6; vertical-align: top; }
    th.sortable { cursor: pointer; user-select: none; }
    tr:hover { background: #f7fbfd; cursor: pointer; }
    .split { display: grid; grid-template-columns: 2fr 1fr; gap: 10px; }
    pre { white-space: pre-wrap; word-break: break-word; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; margin: 0; max-height: 560px; overflow: auto; }
    .muted { color: #5a6778; font-size: 12px; margin: 6px 0 12px; }
    @media (max-width: 900px) {
    .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .filters { grid-template-columns: 1fr; }
      .filters.advanced { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Paper2Data Terminal</h1>
    <div class=\"grid\" id=\"summary\"></div>
    <div class=\"muted\">Use optional advanced filters for domain-specific narrowing.</div>
    <div class=\"filters\">
      <input id=\"q\" placeholder=\"Search title, DOI, methods, findings...\" />
      <select id=\"journal\"><option value=\"\">All journals</option></select>
      <select id=\"repo\"><option value=\"\">All repositories</option></select>
      <button id=\"run\">Run</button>
    </div>
    <div class=\"muted\"><label><input type=\"checkbox\" id=\"show_advanced\" /> Show advanced filters</label></div>
    <div class=\"filters advanced\" id=\"advanced_filters\">
      <select id=\"field\"><option value=\"\">All fields</option></select>
      <select id=\"subcategory\"><option value=\"\">All subcategories</option></select>
      <select id=\"status\"><option value=\"\">All availability</option></select>
      <select id=\"assay\"><option value=\"\">All assay types</option></select>
      <select id=\"organism\"><option value=\"\">All organisms</option></select>
      <input id=\"min_conf\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" placeholder=\"Min conf (0-1)\" />
    </div>
    <div class=\"split\">
      <table>
        <thead>
          <tr>
            <th class=\"sortable\" data-sort=\"title\">Title</th>
            <th class=\"sortable\" data-sort=\"journal\">Journal</th>
            <th class=\"sortable\" data-sort=\"publication_date\">Date</th>
            <th class=\"sortable\" data-sort=\"extraction_confidence\">Conf</th>
          </tr>
        </thead>
        <tbody id=\"rows\"></tbody>
      </table>
      <pre id=\"detail\">Select a paper to view full structured JSON.</pre>
    </div>
    <div class=\"muted\" id=\"status_msg\"></div>
  </div>
  <script>
    let sortBy = 'updated_at';
    let sortDir = 'desc';

    function fillOptions(id, items, placeholder) {
      const el = document.getElementById(id);
      if (!el.multiple) {
        const base = document.createElement('option');
        base.value = '';
        base.textContent = placeholder;
        el.appendChild(base);
      }
      items.forEach(x => {
        const o = document.createElement('option');
        o.value = x.name;
        o.textContent = `${x.name} (${x.count})`;
        el.appendChild(o);
      });
    }

    async function loadSummary() {
      const [statsRes, facetsRes] = await Promise.all([fetch('/api/summary'), fetch('/api/facets')]);
      const stats = await statsRes.json();
      const facets = await facetsRes.json();
      document.getElementById('summary').innerHTML = [
        ['Papers', stats.papers],
        ['Paper Sources', stats.total_sources],
        ['DB Path', stats.db_path]
      ].map(([k,v]) => `<div class=\"card\"><div class=\"k\">${k}</div><div class=\"v\">${v}</div></div>`).join('');

      fillOptions('journal', facets.journals.map(j => ({name: j.journal === 'Unknown' ? '' : j.journal, count: j.count})).filter(x => x.name), 'All journals');
      fillOptions('repo', facets.repositories, 'All repositories');
      fillOptions('status', facets.data_statuses || [], 'All availability');
      fillOptions('assay', facets.assay_types || [], 'All assay types');
      fillOptions('organism', facets.organisms || [], 'All organisms');
      fillOptions('field', facets.field_domains || [], 'All fields');
      fillOptions('subcategory', facets.subcategories || [], 'All subcategories');
    }

    async function runQuery() {
      const params = new URLSearchParams();
      const q = document.getElementById('q').value.trim();
      const field = document.getElementById('field').value;
      const subcategory = document.getElementById('subcategory').value;
      const journal = document.getElementById('journal').value;
      const repo = document.getElementById('repo').value;
      const status = document.getElementById('status').value;
      const assay = document.getElementById('assay').value;
      const organism = document.getElementById('organism').value;
      const minConf = document.getElementById('min_conf').value.trim();

      if (q) params.set('q', q);
      if (field) params.set('field_domain', field);
      if (subcategory) params.set('subcategory', subcategory);
      if (journal) params.set('journal', journal);
      if (repo) params.set('repository', repo);
      if (status) params.set('data_status', status);
      if (assay) params.set('assay', assay);
      if (organism) params.set('organism', organism);
      if (minConf) params.set('min_confidence', minConf);
      params.set('sort_by', sortBy);
      params.set('sort_dir', sortDir);
      params.set('limit', '300');

      const res = await fetch('/api/papers?' + params.toString());
      const rows = await res.json();
      const tbody = document.getElementById('rows');
      const statusMsg = document.getElementById('status_msg');
      tbody.innerHTML = '';
      if (!Array.isArray(rows)) {
        const error = rows && rows.error ? rows.error : 'Unexpected API response';
        statusMsg.textContent = `Query failed: ${error}`;
        tbody.innerHTML = '<tr><td colspan=\"4\">Query error. Check status below.</td></tr>';
        return;
      }
      statusMsg.textContent = `Matched ${rows.length} paper(s). sort=${sortBy} ${sortDir}.`;

      rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.title || ''}</td><td>${row.journal || ''}</td><td>${row.publication_date || ''}</td><td>${(row.extraction_confidence ?? '').toString()}</td>`;
        tr.onclick = async () => {
          const detailRes = await fetch('/api/paper/' + row.paper_id);
          const detail = await detailRes.json();
          document.getElementById('detail').textContent = JSON.stringify(detail.record_json, null, 2);
        };
        tbody.appendChild(tr);
      });

      if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan=\"4\">No papers matched.</td></tr>';
      }
    }

    function applyFieldBehavior() {
      const field = document.getElementById('field').value;
      const assay = document.getElementById('assay');
      const organism = document.getElementById('organism');
      const biologyField = field === '' || field === 'biology';
      assay.disabled = !biologyField;
      organism.disabled = !biologyField;
      if (!biologyField) {
        assay.value = '';
        organism.value = '';
      }
    }

    function initSortHandlers() {
      document.querySelectorAll('th.sortable').forEach(th => {
        th.onclick = () => {
          const col = th.dataset.sort;
          if (sortBy === col) {
            sortDir = sortDir === 'asc' ? 'desc' : 'asc';
          } else {
            sortBy = col;
            sortDir = 'asc';
          }
          runQuery();
        };
      });
    }

    document.getElementById('run').onclick = runQuery;
    document.getElementById('field').onchange = () => {
      applyFieldBehavior();
      runQuery();
    };
    document.getElementById('show_advanced').onchange = (ev) => {
      const panel = document.getElementById('advanced_filters');
      panel.classList.toggle('show', ev.target.checked);
    };
    window.onload = async () => {
      await loadSummary();
      applyFieldBehavior();
      initSortHandlers();
      await runQuery();
    };
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    db: PaperDatabase | None = None

    def _send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.db is None:
            self._send_json({"error": "database unavailable"}, status=500)
            return

        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path == "/api/summary":
            self._send_json(self.db.stats())
            return
        if path == "/api/facets":
            self._send_json(self.db.facet_options())
            return
        if path == "/api/papers":
            q = params.get("q", [""])[0]
            journal = params.get("journal", [""])[0] or None
            repository = params.get("repository", [""])[0] or None
            data_status = params.get("data_status", [""])[0] or None
            assay_type = params.get("assay", [""])[0] or None
            organism = params.get("organism", [""])[0] or None
            field_domain = params.get("field_domain", [""])[0] or None
            subcategory = params.get("subcategory", [""])[0] or None
            min_confidence_raw = params.get("min_confidence", [""])[0]
            limit_raw = params.get("limit", ["100"])[0]
            sort_by = params.get("sort_by", ["updated_at"])[0]
            sort_dir = params.get("sort_dir", ["desc"])[0]
            min_confidence = None
            if min_confidence_raw:
                try:
                    min_confidence = float(min_confidence_raw)
                except ValueError:
                    self._send_json({"error": "min_confidence must be a float"}, status=400)
                    return
            try:
                limit = max(1, min(500, int(limit_raw)))
            except ValueError:
                self._send_json({"error": "limit must be an integer"}, status=400)
                return
            rows = self.db.list_papers(
                q=q,
                journal=journal,
                repository=repository,
                assay_type=assay_type,
                organism=organism,
                field_domain=field_domain,
                subcategory=subcategory,
                data_status=data_status,
                min_confidence=min_confidence,
                sort_by=sort_by,
                sort_dir=sort_dir,
                limit=limit,
            )
            self._send_json(rows)
            return
        if path.startswith("/api/paper/"):
            paper_id = path.split("/", 3)[-1]
            row = self.db.fetch_paper(paper_id)
            if row is None:
                self._send_json({"error": "paper_id not found"}, status=404)
                return
            self._send_json(row)
            return

        self._send_json({"error": "not found"}, status=404)


def run_server(db_path: str, host: str, port: int) -> None:
    db = PaperDatabase(db_path)
    _Handler.db = db
    server = HTTPServer((host, port), _Handler)
    print(f"Paper2Data web UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper2Data local web UI")
    p.add_argument("--db", type=str, default="outputs/paper_terminal.db", help="SQLite DB path")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p


def main() -> None:
    load_env_file()
    args = _build_parser().parse_args()
    run_server(db_path=args.db, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
