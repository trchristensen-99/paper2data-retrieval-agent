from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.database.store import PaperDatabase
from src.utils.env import load_env_file


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paper2Data Terminal</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #1b2635;
      --accent: #0a7ea4;
      --line: #d6dde6;
    }
    body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 14px; font-size: 24px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .card .k { color: #5a6778; font-size: 12px; }
    .card .v { font-size: 21px; font-weight: 700; }
    .filters { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr auto; gap: 8px; margin-bottom: 10px; }
    input, select, button { border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; }
    button { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eef2f6; vertical-align: top; }
    tr:hover { background: #f7fbfd; cursor: pointer; }
    .split { display: grid; grid-template-columns: 2fr 1fr; gap: 10px; }
    pre { white-space: pre-wrap; word-break: break-word; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; margin: 0; max-height: 520px; overflow: auto; }
    @media (max-width: 900px) {
      .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .filters { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Paper2Data Terminal</h1>
    <div class="grid" id="summary"></div>
    <div class="filters">
      <input id="q" placeholder="Search title, DOI, methods, findings..." />
      <select id="journal"><option value="">All journals</option></select>
      <select id="repo"><option value="">All repositories</option></select>
      <input id="min_conf" type="number" min="0" max="1" step="0.01" placeholder="Min conf (0-1)" />
      <button id="run">Run</button>
    </div>
    <div class="split">
      <table>
        <thead><tr><th>Title</th><th>Journal</th><th>Date</th><th>Conf</th><th>Sources</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <pre id="detail">Select a paper to view full structured JSON.</pre>
    </div>
  </div>
  <script>
    async function loadSummary() {
      const [statsRes, facetsRes] = await Promise.all([fetch('/api/summary'), fetch('/api/facets')]);
      const stats = await statsRes.json();
      const facets = await facetsRes.json();
      document.getElementById('summary').innerHTML = [
        ['Papers', stats.papers],
        ['Versions', stats.versions],
        ['Total Sources', stats.total_sources],
        ['DB Path', stats.db_path]
      ].map(([k,v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
      const journal = document.getElementById('journal');
      facets.journals.forEach(j => {
        const o = document.createElement('option');
        o.value = j.journal === 'Unknown' ? '' : j.journal;
        o.textContent = `${j.journal} (${j.count})`;
        journal.appendChild(o);
      });
      const repo = document.getElementById('repo');
      facets.repositories.forEach(r => {
        const o = document.createElement('option');
        o.value = r.name;
        o.textContent = `${r.name} (${r.count})`;
        repo.appendChild(o);
      });
    }

    async function runQuery() {
      const params = new URLSearchParams();
      const q = document.getElementById('q').value.trim();
      const journal = document.getElementById('journal').value;
      const repo = document.getElementById('repo').value;
      const minConf = document.getElementById('min_conf').value.trim();
      if (q) params.set('q', q);
      if (journal) params.set('journal', journal);
      if (repo) params.set('repository', repo);
      if (minConf) params.set('min_confidence', minConf);
      params.set('limit', '200');
      const res = await fetch('/api/papers?' + params.toString());
      const rows = await res.json();
      const tbody = document.getElementById('rows');
      tbody.innerHTML = '';
      rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.title || ''}</td><td>${row.journal || ''}</td><td>${row.publication_date || ''}</td><td>${(row.extraction_confidence ?? '').toString()}</td><td>${row.source_count}</td>`;
        tr.onclick = async () => {
          const detailRes = await fetch('/api/paper/' + row.paper_id);
          const detail = await detailRes.json();
          document.getElementById('detail').textContent = JSON.stringify(detail.record_json, null, 2);
        };
        tbody.appendChild(tr);
      });
      if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5">No papers matched.</td></tr>';
      }
    }

    document.getElementById('run').onclick = runQuery;
    window.onload = async () => {
      await loadSummary();
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
            min_confidence_raw = params.get("min_confidence", [""])[0]
            limit_raw = params.get("limit", ["100"])[0]
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
                min_confidence=min_confidence,
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
    server = ThreadingHTTPServer((host, port), _Handler)
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
