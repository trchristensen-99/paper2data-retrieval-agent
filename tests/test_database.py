from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.database.harmonizer import HarmonizationOutput
from src.database.store import PaperDatabase
from src.schemas.models import (
    DataAvailabilityReport,
    DescriptiveStat,
    MetadataRecord,
    MethodsSummary,
    MethodBenchmark,
    PaperRecord,
    ResultsSummary,
)


def _record(title: str, doi: str | None = None, pmid: str | None = None) -> PaperRecord:
    return PaperRecord(
        metadata=MetadataRecord(
            title=title,
            authors=["A. Author"],
            doi=doi,
            pmid=pmid,
            journal="Sci Data",
            publication_date="2025-01-01",
            paper_type="dataset_descriptor",
        ),
        methods=MethodsSummary(
            organisms=["Homo sapiens"],
            cell_types=[],
            assay_types=["RNA-seq"],
            sample_sizes={"n": "10"},
            statistical_tests=["t-test"],
            experimental_design="case-control",
            methods_completeness="adequate",
        ),
        results=ResultsSummary(
            paper_type="dataset_descriptor",
            dataset_properties=[DescriptiveStat(property="records", value="10", context="baseline")],
            synthesized_claims=["baseline"],
            method_benchmarks=[MethodBenchmark(task="none", metric="none", value="n/a", context="baseline")],
        ),
        data_accessions=[],
        data_availability=DataAvailabilityReport(
            overall_status="accessible",
            claimed_repositories=["Zenodo"],
            verified_repositories=["Zenodo"],
            discrepancies=[],
            notes="ok",
        ),
        extraction_timestamp="2026-02-12T00:00:00",
        extraction_confidence=0.8,
    )


@pytest.mark.asyncio
async def test_db_insert_and_query(tmp_path: Path) -> None:
    db = PaperDatabase(str(tmp_path / "paper.db"))
    try:
        result = await db.upsert_record(_record("Test Paper", doi="10.1000/test"))
        assert result.action == "inserted"
        rows = db.search("Test Paper")
        assert len(rows) == 1
        assert rows[0]["doi"] == "10.1000/test"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_db_upsert_merges_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def _fake_harmonize(existing: PaperRecord, incoming: PaperRecord) -> HarmonizationOutput:
        merged = existing.model_copy(deep=True)
        merged.metadata.keywords = ["merged"]
        merged.extraction_confidence = 0.9
        return HarmonizationOutput(
            merged_record=merged,
            summary="merged",
            conflicts_resolved=["keywords"],
            unresolved_conflicts=[],
            confidence=0.9,
        )

    monkeypatch.setattr("src.database.store.harmonize_records", _fake_harmonize)

    db = PaperDatabase(str(tmp_path / "paper.db"))
    try:
        one = _record("Same Title", doi="10.1000/same")
        two = _record("Same Title", doi="10.1000/same")
        await db.upsert_record(one)
        res = await db.upsert_record(two)
        assert res.action == "updated"
        stats = db.stats()
        assert stats["papers"] == 1
        row = db.search("Same Title")[0]
        assert row["source_count"] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_ingest_structured_file(tmp_path: Path) -> None:
    db = PaperDatabase(str(tmp_path / "paper.db"))
    try:
        path = tmp_path / "structured_record.json"
        path.write_text(json.dumps(_record("From File", pmid="1234").model_dump()), encoding="utf-8")
        res = await db.ingest_structured_record_file(path)
        assert res.action == "inserted"
        assert db.search("From File")[0]["pmid"] == "1234"
    finally:
        db.close()
