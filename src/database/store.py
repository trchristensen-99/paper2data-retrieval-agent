from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.database.harmonizer import harmonize_records
from src.schemas.models import PaperRecord


def _norm_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True)


def _now() -> str:
    return datetime.utcnow().isoformat()


def compute_paper_key(record: PaperRecord) -> str:
    if record.metadata.doi:
        return f"doi:{record.metadata.doi.strip().lower()}"
    if record.metadata.pmid:
        return f"pmid:{record.metadata.pmid.strip()}"
    title = _norm_text(record.metadata.title)
    return "title:" + hashlib.sha1(title.encode("utf-8")).hexdigest()


@dataclass
class UpsertResult:
    paper_id: str
    action: str
    merged: bool


class PaperDatabase:
    def __init__(self, db_path: str = "outputs/paper_terminal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                canonical_key TEXT NOT NULL,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                doi TEXT,
                pmid TEXT,
                journal TEXT,
                publication_date TEXT,
                extraction_confidence REAL,
                source_count INTEGER NOT NULL DEFAULT 1,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(normalized_title)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                source_path TEXT,
                extraction_timestamp TEXT,
                extraction_confidence REAL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_search (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                journal TEXT,
                keywords TEXT,
                methods TEXT,
                findings TEXT,
                repositories TEXT,
                assay_types TEXT,
                organisms TEXT,
                data_status TEXT,
                FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
            )
            """
        )
        self._ensure_column("paper_search", "assay_types", "TEXT")
        self._ensure_column("paper_search", "organisms", "TEXT")
        self._ensure_column("paper_search", "data_status", "TEXT")
        self.conn.commit()

    def _table_columns(self, table: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r["name"]) for r in rows}

    def _ensure_column(self, table: str, column: str, sql_type: str) -> None:
        if column not in self._table_columns(table):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    def _find_existing(self, record: PaperRecord) -> sqlite3.Row | None:
        cur = self.conn.cursor()
        if record.metadata.doi:
            row = cur.execute("SELECT * FROM papers WHERE doi = ?", (record.metadata.doi,)).fetchone()
            if row:
                return row
        if record.metadata.pmid:
            row = cur.execute("SELECT * FROM papers WHERE pmid = ?", (record.metadata.pmid,)).fetchone()
            if row:
                return row
        normalized_title = _norm_text(record.metadata.title)
        return cur.execute(
            "SELECT * FROM papers WHERE normalized_title = ?", (normalized_title,)
        ).fetchone()

    def _update_search_row(self, paper_id: str, record: PaperRecord) -> None:
        authors = "; ".join(record.metadata.authors)
        keywords = "; ".join(record.metadata.keywords)
        methods = " | ".join(
            [
                record.methods.experimental_design,
                "; ".join(record.methods.assay_types),
                "; ".join(record.methods.organisms),
                "; ".join(record.methods.cell_types),
                "; ".join(record.methods.statistical_tests),
            ]
        )
        findings = " | ".join(
            [f"{f.claim} {f.metric} {f.value}" for f in record.results.quantitative_findings]
            + record.results.qualitative_findings
        )
        repositories = "; ".join(
            sorted({x.repository for x in record.data_accessions} | set(record.code_repositories))
        )
        assay_types = "; ".join(record.methods.assay_types)
        organisms = "; ".join(record.methods.organisms)
        data_status = record.data_availability.overall_status
        self.conn.execute(
            """
            INSERT INTO paper_search
            (paper_id, title, authors, journal, keywords, methods, findings, repositories, assay_types, organisms, data_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                journal=excluded.journal,
                keywords=excluded.keywords,
                methods=excluded.methods,
                findings=excluded.findings,
                repositories=excluded.repositories,
                assay_types=excluded.assay_types,
                organisms=excluded.organisms,
                data_status=excluded.data_status
            """,
            (
                paper_id,
                record.metadata.title,
                authors,
                record.metadata.journal,
                keywords,
                methods,
                findings,
                repositories,
                assay_types,
                organisms,
                data_status,
            ),
        )

    def _insert_version(self, paper_id: str, source_path: str | None, record: PaperRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_versions
                (paper_id, source_path, extraction_timestamp, extraction_confidence, record_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                source_path,
                record.extraction_timestamp,
                record.extraction_confidence,
                _safe_json(record.model_dump()),
                _now(),
            ),
        )

    async def upsert_record(self, record: PaperRecord, source_path: str | None = None) -> UpsertResult:
        existing = self._find_existing(record)
        if existing is None:
            paper_id = hashlib.sha1((compute_paper_key(record) + _now()).encode("utf-8")).hexdigest()[:16]
            self.conn.execute(
                """
                INSERT INTO papers
                    (paper_id, canonical_key, title, normalized_title, doi, pmid, journal,
                     publication_date, extraction_confidence, source_count, record_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    compute_paper_key(record),
                    record.metadata.title,
                    _norm_text(record.metadata.title),
                    record.metadata.doi,
                    record.metadata.pmid,
                    record.metadata.journal,
                    record.metadata.publication_date,
                    record.extraction_confidence,
                    1,
                    _safe_json(record.model_dump()),
                    _now(),
                    _now(),
                ),
            )
            self._insert_version(paper_id, source_path, record)
            self._update_search_row(paper_id, record)
            self.conn.commit()
            return UpsertResult(paper_id=paper_id, action="inserted", merged=False)

        paper_id = existing["paper_id"]
        existing_record = PaperRecord.model_validate(json.loads(existing["record_json"]))
        harmonized = await harmonize_records(existing_record, record)
        merged = harmonized.merged_record

        self.conn.execute(
            """
            UPDATE papers
               SET canonical_key = ?,
                   title = ?,
                   normalized_title = ?,
                   doi = ?,
                   pmid = ?,
                   journal = ?,
                   publication_date = ?,
                   extraction_confidence = ?,
                   source_count = source_count + 1,
                   record_json = ?,
                   updated_at = ?
             WHERE paper_id = ?
            """,
            (
                compute_paper_key(merged),
                merged.metadata.title,
                _norm_text(merged.metadata.title),
                merged.metadata.doi,
                merged.metadata.pmid,
                merged.metadata.journal,
                merged.metadata.publication_date,
                merged.extraction_confidence,
                _safe_json(merged.model_dump()),
                _now(),
                paper_id,
            ),
        )
        self._insert_version(paper_id, source_path, record)
        self._update_search_row(paper_id, merged)
        self.conn.commit()
        return UpsertResult(paper_id=paper_id, action="updated", merged=True)

    async def ingest_structured_record_file(self, path: Path) -> UpsertResult:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = PaperRecord.model_validate(payload)
        return await self.upsert_record(record, source_path=str(path))

    async def ingest_many(self, paths: Iterable[Path]) -> list[UpsertResult]:
        out: list[UpsertResult] = []
        for p in paths:
            out.append(await self.ingest_structured_record_file(p))
        return out

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        like = f"%{query}%"
        rows = self.conn.execute(
            """
            SELECT p.paper_id, p.title, p.doi, p.pmid, p.journal, p.publication_date,
                   p.extraction_confidence, p.source_count
              FROM papers p
              LEFT JOIN paper_search s ON s.paper_id = p.paper_id
             WHERE p.title LIKE ?
                OR p.doi LIKE ?
                OR p.pmid LIKE ?
                OR s.authors LIKE ?
                OR s.keywords LIKE ?
                OR s.methods LIKE ?
                OR s.findings LIKE ?
                OR s.repositories LIKE ?
             ORDER BY p.updated_at DESC
             LIMIT ?
            """,
            (like, like, like, like, like, like, like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_papers(
        self,
        *,
        q: str = "",
        journal: str | None = None,
        repository: str | None = None,
        assay_types: list[str] | None = None,
        organisms: list[str] | None = None,
        data_status: str | None = None,
        min_confidence: float | None = None,
        sort_by: str = "updated_at",
        sort_dir: str = "desc",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if q.strip():
            like = f"%{q.strip()}%"
            clauses.append(
                """(
                    p.title LIKE ?
                    OR p.doi LIKE ?
                    OR p.pmid LIKE ?
                    OR s.authors LIKE ?
                    OR s.keywords LIKE ?
                    OR s.methods LIKE ?
                    OR s.findings LIKE ?
                    OR s.repositories LIKE ?
                )"""
            )
            params.extend([like, like, like, like, like, like, like, like])
        if journal:
            clauses.append("p.journal = ?")
            params.append(journal)
        if repository:
            clauses.append("s.repositories LIKE ?")
            params.append(f"%{repository}%")
        if assay_types:
            assay_sub = []
            for value in assay_types:
                assay_sub.append("s.assay_types LIKE ?")
                params.append(f"%{value}%")
            clauses.append("(" + " OR ".join(assay_sub) + ")")
        if organisms:
            org_sub = []
            for value in organisms:
                org_sub.append("s.organisms LIKE ?")
                params.append(f"%{value}%")
            clauses.append("(" + " OR ".join(org_sub) + ")")
        if data_status:
            clauses.append("s.data_status = ?")
            params.append(data_status)
        if min_confidence is not None:
            clauses.append("p.extraction_confidence >= ?")
            params.append(min_confidence)

        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

        sort_cols = {
            "title": "p.title",
            "journal": "p.journal",
            "publication_date": "p.publication_date",
            "extraction_confidence": "p.extraction_confidence",
            "source_count": "p.source_count",
            "updated_at": "p.updated_at",
        }
        sort_col = sort_cols.get(sort_by, "p.updated_at")
        direction = "ASC" if sort_dir.lower() == "asc" else "DESC"

        rows = self.conn.execute(
            f"""
            SELECT p.paper_id, p.title, p.doi, p.pmid, p.journal, p.publication_date,
                   p.extraction_confidence, p.source_count,
                   COALESCE(s.repositories, '') AS repositories,
                   COALESCE(s.assay_types, '') AS assay_types,
                   COALESCE(s.organisms, '') AS organisms,
                   COALESCE(s.data_status, '') AS data_status
              FROM papers p
              LEFT JOIN paper_search s ON s.paper_id = p.paper_id
              {where_sql}
             ORDER BY {sort_col} {direction}
             LIMIT ?
            """,
            tuple(params + [limit]),
        ).fetchall()
        return [dict(r) for r in rows]

    def facet_options(self) -> dict[str, Any]:
        journal_rows = self.conn.execute(
            """
            SELECT COALESCE(journal, 'Unknown') AS journal, COUNT(*) AS count
              FROM papers
             GROUP BY COALESCE(journal, 'Unknown')
             ORDER BY count DESC, journal ASC
            """
        ).fetchall()
        repo_rows = self.conn.execute(
            "SELECT repositories, assay_types, organisms, data_status FROM paper_search"
        ).fetchall()
        repo_counts: dict[str, int] = {}
        assay_counts: dict[str, int] = {}
        organism_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for row in repo_rows:
            for token in str(row["repositories"] or "").split(";"):
                repo = token.strip()
                if not repo:
                    continue
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
            for token in str(row["assay_types"] or "").split(";"):
                assay = token.strip()
                if not assay:
                    continue
                assay_counts[assay] = assay_counts.get(assay, 0) + 1
            for token in str(row["organisms"] or "").split(";"):
                organism = token.strip()
                if not organism:
                    continue
                organism_counts[organism] = organism_counts.get(organism, 0) + 1
            status = str(row["data_status"] or "").strip()
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
        repositories = [
            {"name": key, "count": value}
            for key, value in sorted(repo_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        assay_types = [
            {"name": key, "count": value}
            for key, value in sorted(assay_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        organisms = [
            {"name": key, "count": value}
            for key, value in sorted(organism_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        data_statuses = [
            {"name": key, "count": value}
            for key, value in sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        return {
            "journals": [dict(r) for r in journal_rows],
            "repositories": repositories,
            "assay_types": assay_types,
            "organisms": organisms,
            "data_statuses": data_statuses,
        }

    def fetch_paper(self, paper_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["record_json"] = json.loads(data["record_json"])
        return data

    def stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) as papers, COALESCE(SUM(source_count),0) as total_sources FROM papers"
        ).fetchone()
        versions = self.conn.execute("SELECT COUNT(*) as n FROM paper_versions").fetchone()["n"]
        return {
            "papers": row["papers"],
            "total_sources": row["total_sources"],
            "versions": versions,
            "db_path": str(self.db_path),
        }
