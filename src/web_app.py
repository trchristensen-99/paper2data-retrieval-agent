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
    .grid { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .card .k { color: #5a6778; font-size: 12px; }
    .card .v { font-size: 21px; font-weight: 700; }
    .filters { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr auto; gap: 8px; margin-bottom: 8px; }
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
    .detail-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; min-height: 560px; }
    .detail-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
    .detail-tab { background: #eef5f8; color: var(--ink); border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; cursor: pointer; font-size: 12px; }
    .detail-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .detail-tools { display: grid; grid-template-columns: 1fr; margin-bottom: 8px; }
    .detail-table { width: 100%; border-collapse: collapse; }
    .detail-table th, .detail-table td { border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; padding: 6px; }
    .detail-table th { width: 36%; color: #435366; }
    .subtable-wrap { border: 1px solid #e7eef5; border-radius: 8px; max-height: 280px; overflow: auto; background: #fff; }
    .subtable { width: 100%; border-collapse: collapse; }
    .subtable th, .subtable td { border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; padding: 6px; font-size: 12px; }
    .subtable th { position: sticky; top: 0; background: #f7fbfd; z-index: 1; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef5f8; border: 1px solid var(--line); font-size: 11px; margin-right: 4px; margin-bottom: 4px; }
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
    <div class=\"muted\">Use field/subfield in main filters. Repositories and assay-level filters are in advanced filters.</div>
    <div class=\"filters\">
      <input id=\"q\" placeholder=\"Search title, DOI, methods, findings...\" />
      <select id=\"field\"><option value=\"\">All fields</option></select>
      <select id=\"subfield\"><option value=\"\">All subfields</option></select>
      <select id=\"journal\"><option value=\"\">All journals</option></select>
      <button id=\"run\">Run</button>
    </div>
    <div class=\"muted\"><label><input type=\"checkbox\" id=\"show_advanced\" /> Show advanced filters</label></div>
    <div class=\"filters advanced\" id=\"advanced_filters\">
      <select id=\"paper_type\"><option value=\"\">All paper types</option></select>
      <select id=\"repo\"><option value=\"\">All repositories</option></select>
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
      <div id=\"detail\" class=\"detail-panel\">Select a paper to view structured details.</div>
    </div>
    <div class=\"muted\" id=\"status_msg\"></div>
  </div>
  <script>
    let sortBy = 'updated_at';
    let sortDir = 'desc';
    let fieldSubfields = {};
    let detailTab = 'overview';
    let detailFilter = '';
    let currentRecord = null;

    function fillOptions(id, items, placeholder) {
      const el = document.getElementById(id);
      el.innerHTML = '';
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

    function fillSubcategoryOptions() {
      const field = document.getElementById('field').value;
      const list = field && fieldSubfields[field] ? fieldSubfields[field] : [];
      const selected = document.getElementById('subfield').value;
      fillOptions('subfield', list, 'All subfields');
      if (list.some(x => x.name === selected)) {
        document.getElementById('subfield').value = selected;
      }
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('\"', '&quot;')
        .replaceAll(\"'\", '&#39;');
    }

    function valueToText(value) {
      if (value === null || value === undefined) return '';
      if (Array.isArray(value)) return value.map(v => valueToText(v)).filter(Boolean).join('; ');
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    }

    function buildDetailRows(record) {
      const metadata = record.metadata || {};
      const methods = record.methods || {};
      const results = record.results || {};
      const dataAvailability = record.data_availability || {};
      const accessions = Array.isArray(record.data_accessions) ? record.data_accessions : [];
      const experimental = Array.isArray(results.experimental_findings) ? results.experimental_findings : [];

      return {
        overview: [
          ['Title', metadata.title || ''],
          ['Authors', valueToText(metadata.authors || [])],
          ['Paper Type', metadata.paper_type || (results.paper_type || '')],
          ['Journal/Venue', metadata.journal || ''],
          ['Publication Date', metadata.publication_date || ''],
          ['DOI', metadata.doi || ''],
          ['PMID', metadata.pmid || ''],
          ['License', metadata.license || ''],
          ['Field', metadata.category || ''],
          ['Subfield', metadata.subcategory || ''],
          ['Confidence', Number(record.extraction_confidence || 0).toFixed(2)],
        ],
        metadata: [
          ['Keywords', valueToText(metadata.keywords || [])],
          ['Funding Sources', valueToText(metadata.funding_sources || [])],
          ['Conflicts of Interest', metadata.conflicts_of_interest || ''],
        ],
        methods: [
          ['Organisms', valueToText(methods.organisms || [])],
          ['Cell Types', valueToText(methods.cell_types || [])],
          ['Assay Types', valueToText(methods.assay_types || [])],
          ['Sample Sizes', valueToText(methods.sample_sizes || {})],
          ['Statistical Tests', valueToText(methods.statistical_tests || [])],
          ['Experimental Design', methods.experimental_design || ''],
          ['Methods Completeness', methods.methods_completeness || ''],
        ],
        results: [
          ['Result Mode', results.paper_type || metadata.paper_type || ''],
          ['Spin Assessment', results.spin_assessment || ''],
          ['Synthesized Claims Count', Array.isArray(results.synthesized_claims) ? results.synthesized_claims.length : 0],
          ['Experimental Findings Count', experimental.length],
          ['Dataset Properties Count', Array.isArray(results.dataset_properties) ? results.dataset_properties.length : 0],
          ['Method Benchmarks Count', Array.isArray(results.method_benchmarks) ? results.method_benchmarks.length : 0],
        ],
        data: [
          ['Availability Status', dataAvailability.overall_status || ''],
          ['Claimed Repositories', valueToText(dataAvailability.claimed_repositories || [])],
          ['Verified Repositories', valueToText(dataAvailability.verified_repositories || [])],
          ['Discrepancies', valueToText(dataAvailability.discrepancies || [])],
          ['Data Availability Notes', dataAvailability.notes || ''],
          ['Code Repositories', valueToText(record.code_repositories || [])],
          ['Data Accessions Count', accessions.length],
        ],
      };
    }

    function renderDetailTable(rows) {
      const f = detailFilter.trim().toLowerCase();
      const filtered = rows.filter(([k, v]) => {
        if (!f) return true;
        const text = `${k} ${valueToText(v)}`.toLowerCase();
        return text.includes(f);
      });
      if (!filtered.length) {
        return '<div class=\"muted\">No matching fields in this section.</div>';
      }
      return `<table class=\"detail-table\"><tbody>${
        filtered.map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(valueToText(v))}</td></tr>`).join('')
      }</tbody></table>`;
    }

    function renderPills(items) {
      const values = Array.isArray(items) ? items.map(x => valueToText(x)).filter(Boolean) : [];
      if (!values.length) return '<div class=\"muted\">None</div>';
      return `<div>${values.map(v => `<span class=\"pill\">${escapeHtml(v)}</span>`).join('')}</div>`;
    }

    function renderFindingsTable(record) {
      const findings = ((record.results || {}).experimental_findings || []);
      const q = detailFilter.trim().toLowerCase();
      const filtered = findings.filter(f => {
        if (!q) return true;
        return [
          f.claim, f.metric, f.value, f.effect_size, f.confidence_interval, f.p_value, f.context, f.confidence
        ].map(valueToText).join(' ').toLowerCase().includes(q);
      });
      if (!filtered.length) return '<div class=\"muted\">No matching experimental findings.</div>';
      return `
        <div class=\"subtable-wrap\">
          <table class=\"subtable\">
            <thead>
              <tr><th>#</th><th>Claim</th><th>Metric</th><th>Value</th><th>Effect Size</th><th>CI</th><th>p</th><th>Context</th><th>Conf</th></tr>
            </thead>
            <tbody>
              ${filtered.map((f, i) => `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(valueToText(f.claim))}</td>
                <td>${escapeHtml(valueToText(f.metric))}</td>
                <td>${escapeHtml(valueToText(f.value))}</td>
                <td>${escapeHtml(valueToText(f.effect_size))}</td>
                <td>${escapeHtml(valueToText(f.confidence_interval))}</td>
                <td>${escapeHtml(valueToText(f.p_value))}</td>
                <td>${escapeHtml(valueToText(f.context))}</td>
                <td>${escapeHtml(valueToText(f.confidence))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderDatasetPropertiesTable(record) {
      const props = ((record.results || {}).dataset_properties || []);
      const q = detailFilter.trim().toLowerCase();
      const filtered = props.filter(p => {
        if (!q) return true;
        return [p.property, p.value, p.context].map(valueToText).join(' ').toLowerCase().includes(q);
      });
      if (!filtered.length) return '<div class=\"muted\">No matching dataset properties.</div>';
      return `
        <div class=\"subtable-wrap\">
          <table class=\"subtable\">
            <thead><tr><th>#</th><th>Property</th><th>Value</th><th>Context</th></tr></thead>
            <tbody>
              ${filtered.map((p, i) => `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(valueToText(p.property))}</td>
                <td>${escapeHtml(valueToText(p.value))}</td>
                <td>${escapeHtml(valueToText(p.context))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderMethodBenchmarksTable(record) {
      const rows = ((record.results || {}).method_benchmarks || []);
      const q = detailFilter.trim().toLowerCase();
      const filtered = rows.filter(b => {
        if (!q) return true;
        return [b.task, b.metric, b.value, b.baseline, b.context].map(valueToText).join(' ').toLowerCase().includes(q);
      });
      if (!filtered.length) return '<div class=\"muted\">No matching method benchmarks.</div>';
      return `
        <div class=\"subtable-wrap\">
          <table class=\"subtable\">
            <thead><tr><th>#</th><th>Task</th><th>Metric</th><th>Value</th><th>Baseline</th><th>Context</th></tr></thead>
            <tbody>
              ${filtered.map((b, i) => `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(valueToText(b.task))}</td>
                <td>${escapeHtml(valueToText(b.metric))}</td>
                <td>${escapeHtml(valueToText(b.value))}</td>
                <td>${escapeHtml(valueToText(b.baseline))}</td>
                <td>${escapeHtml(valueToText(b.context))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderAccessionsTable(record) {
      const accessions = record.data_accessions || [];
      const q = detailFilter.trim().toLowerCase();
      const filtered = accessions.filter(a => {
        if (!q) return true;
        return [
          a.repository, a.accession_id, a.url, a.category, a.data_format, a.file_count, a.is_accessible, a.description
        ].map(valueToText).join(' ').toLowerCase().includes(q);
      });
      if (!filtered.length) return '<div class=\"muted\">No matching accessions.</div>';
      return `
        <div class=\"subtable-wrap\">
          <table class=\"subtable\">
            <thead>
              <tr><th>#</th><th>Repository</th><th>Accession</th><th>URL</th><th>Category</th><th>Format</th><th>Files</th><th>Accessible</th><th>Description</th></tr>
            </thead>
            <tbody>
              ${filtered.map((a, i) => `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(valueToText(a.repository))}</td>
                <td>${escapeHtml(valueToText(a.accession_id))}</td>
                <td>${escapeHtml(valueToText(a.url))}</td>
                <td>${escapeHtml(valueToText(a.category))}</td>
                <td>${escapeHtml(valueToText(a.data_format))}</td>
                <td>${escapeHtml(valueToText(a.file_count))}</td>
                <td>${escapeHtml(valueToText(a.is_accessible))}</td>
                <td>${escapeHtml(valueToText(a.description))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderResultsTab(record, rows) {
      const synthesized = ((record.results || {}).synthesized_claims || []);
      return `
        ${renderDetailTable(rows)}
        <div class=\"muted\">Synthesized claims</div>
        ${renderPills(synthesized)}
        <div class=\"muted\" style=\"margin-top:8px;\">Experimental findings (filter applies here)</div>
        ${renderFindingsTable(record)}
        <div class=\"muted\" style=\"margin-top:8px;\">Dataset properties (filter applies here)</div>
        ${renderDatasetPropertiesTable(record)}
        <div class=\"muted\" style=\"margin-top:8px;\">Method benchmarks (filter applies here)</div>
        ${renderMethodBenchmarksTable(record)}
      `;
    }

    function renderDataTab(record, rows) {
      return `
        ${renderDetailTable(rows)}
        <div class=\"muted\" style=\"margin-top:8px;\">Data accessions (filter applies here)</div>
        ${renderAccessionsTable(record)}
      `;
    }

    function renderDetailPanel() {
      const detailEl = document.getElementById('detail');
      if (!currentRecord) {
        detailEl.textContent = 'Select a paper to view structured details.';
        return;
      }

      const tabs = [
        ['overview', 'Overview'],
        ['metadata', 'Metadata'],
        ['methods', 'Methods'],
        ['results', 'Results'],
        ['data', 'Data'],
        ['raw', 'Raw JSON'],
      ];
      const rowsByTab = buildDetailRows(currentRecord);
      const isRaw = detailTab === 'raw';
      let body = '';
      if (isRaw) {
        body = `<pre>${escapeHtml(JSON.stringify(currentRecord, null, 2))}</pre>`;
      } else if (detailTab === 'results') {
        body = renderResultsTab(currentRecord, rowsByTab.results || []);
      } else if (detailTab === 'data') {
        body = renderDataTab(currentRecord, rowsByTab.data || []);
      } else {
        body = renderDetailTable(rowsByTab[detailTab] || []);
      }

      detailEl.innerHTML = `
        <div class=\"detail-tabs\">
          ${tabs.map(([id, label]) => `<button class=\"detail-tab ${id === detailTab ? 'active' : ''}\" data-tab=\"${id}\">${label}</button>`).join('')}
        </div>
        <div class=\"detail-tools\">
          ${isRaw ? '' : '<input id=\"detail_filter\" placeholder=\"Filter this section...\" />'}
        </div>
        ${body}
      `;

      detailEl.querySelectorAll('.detail-tab').forEach(btn => {
        btn.onclick = () => {
          detailTab = btn.dataset.tab || 'overview';
          detailFilter = '';
          renderDetailPanel();
        };
      });
      const filterEl = document.getElementById('detail_filter');
      if (filterEl) {
        filterEl.value = detailFilter;
        filterEl.oninput = (ev) => {
          detailFilter = ev.target.value || '';
          renderDetailPanel();
        };
      }
    }

    async function loadSummary() {
      const [statsRes, facetsRes] = await Promise.all([fetch('/api/summary'), fetch('/api/facets')]);
      const stats = await statsRes.json();
      const facets = await facetsRes.json();
      fieldSubfields = facets.field_subfields || facets.category_subcategories || {};
      document.getElementById('summary').innerHTML = [
        ['Papers', stats.papers],
        ['DB Path', stats.db_path]
      ].map(([k,v]) => `<div class=\"card\"><div class=\"k\">${k}</div><div class=\"v\">${v}</div></div>`).join('');

      fillOptions('journal', facets.journals.map(j => ({name: j.journal === 'Unknown' ? '' : j.journal, count: j.count})).filter(x => x.name), 'All journals');
      fillOptions('repo', facets.repositories, 'All repositories');
      fillOptions('status', facets.data_statuses || [], 'All availability');
      fillOptions('paper_type', facets.paper_types || [], 'All paper types');
      fillOptions('assay', facets.assay_types || [], 'All assay types');
      fillOptions('organism', facets.organisms || [], 'All organisms');
      fillOptions('field', facets.fields || facets.field_domains || [], 'All fields');
      fillSubcategoryOptions();
    }

    async function runQuery() {
      const params = new URLSearchParams();
      const q = document.getElementById('q').value.trim();
      const field = document.getElementById('field').value;
      const subfield = document.getElementById('subfield').value;
      const journal = document.getElementById('journal').value;
      const paperType = document.getElementById('paper_type').value;
      const repo = document.getElementById('repo').value;
      const status = document.getElementById('status').value;
      const assay = document.getElementById('assay').value;
      const organism = document.getElementById('organism').value;
      const minConf = document.getElementById('min_conf').value.trim();

      if (q) params.set('q', q);
      if (field) params.set('field_domain', field);
      if (subfield) params.set('subfield', subfield);
      if (journal) params.set('journal', journal);
      if (paperType) params.set('paper_type', paperType);
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
        const conf = row.extraction_confidence === null || row.extraction_confidence === undefined
          ? ''
          : Number(row.extraction_confidence).toFixed(2);
        tr.innerHTML = `<td>${row.title || ''}</td><td>${row.journal || ''}</td><td>${row.publication_date || ''}</td><td>${conf}</td>`;
        tr.onclick = async () => {
          const detailRes = await fetch('/api/paper/' + row.paper_id);
          const detail = await detailRes.json();
          currentRecord = detail.record_json || null;
          detailTab = 'overview';
          detailFilter = '';
          renderDetailPanel();
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
      fillSubcategoryOptions();
      applyFieldBehavior();
      runQuery();
    };
    document.getElementById('subfield').onchange = runQuery;
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
            paper_type = params.get("paper_type", [""])[0] or None
            repository = params.get("repository", [""])[0] or None
            data_status = params.get("data_status", [""])[0] or None
            assay_type = params.get("assay", [""])[0] or None
            organism = params.get("organism", [""])[0] or None
            field_domain = params.get("field_domain", [""])[0] or None
            subfield = params.get("subfield", [""])[0] or None
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
                paper_type=paper_type,
                repository=repository,
                assay_type=assay_type,
                organism=organism,
                field_domain=field_domain,
                subfield=subfield,
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
