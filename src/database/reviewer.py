from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.agents.manager import run_pipeline
from src.database.harmonizer import harmonize_records
from src.database.store import PaperDatabase
from src.schemas.models import PaperRecord

ALLOWED_SECTIONS = {
    "metadata",
    "methods",
    "results",
    "data_accessions",
    "data_availability",
    "code_repositories",
}


@dataclass
class ReviewUpdateResult:
    paper_id: str
    sections_updated: list[str]
    changed_top_level_fields: list[str]
    confidence_before: float
    confidence_after: float


def _selective_merge(
    existing: PaperRecord,
    reviewed: PaperRecord,
    sections: set[str],
) -> PaperRecord:
    merged = existing.model_copy(deep=True)
    if "metadata" in sections:
        merged.metadata = reviewed.metadata
    if "methods" in sections:
        merged.methods = reviewed.methods
    if "results" in sections:
        merged.results = reviewed.results
    if "data_accessions" in sections:
        merged.data_accessions = reviewed.data_accessions
    if "data_availability" in sections:
        merged.data_availability = reviewed.data_availability
    if "code_repositories" in sections:
        merged.code_repositories = reviewed.code_repositories

    merged.extraction_confidence = reviewed.extraction_confidence
    merged.extraction_timestamp = reviewed.extraction_timestamp
    return merged


def _changed_top_level_fields(before: PaperRecord, after: PaperRecord) -> list[str]:
    b = before.model_dump()
    a = after.model_dump()
    changed: list[str] = []
    for key in ("metadata", "methods", "results", "data_accessions", "data_availability", "code_repositories"):
        if b.get(key) != a.get(key):
            changed.append(key)
    return changed


def parse_sections(raw: str | None) -> set[str]:
    if not raw or raw.strip().lower() == "all":
        return set(ALLOWED_SECTIONS)
    sections = {x.strip() for x in raw.split(",") if x.strip()}
    invalid = sorted(sections - ALLOWED_SECTIONS)
    if invalid:
        raise ValueError(f"Invalid sections: {invalid}. Allowed: {sorted(ALLOWED_SECTIONS)}")
    return sections


async def review_and_update_entry(
    *,
    db: PaperDatabase,
    paper_id: str,
    paper_markdown_path: Path,
    sections: Iterable[str] | None = None,
) -> ReviewUpdateResult:
    existing = db.fetch_paper_record(paper_id)
    if existing is None:
        raise ValueError(f"paper_id not found: {paper_id}")
    if not paper_markdown_path.exists():
        raise FileNotFoundError(f"Paper markdown file not found: {paper_markdown_path}")

    paper_markdown = paper_markdown_path.read_text(encoding="utf-8")
    new_artifacts = await run_pipeline(paper_markdown)
    reviewed = new_artifacts.record

    harmonized = await harmonize_records(existing, reviewed)
    reviewed_merged = harmonized.merged_record

    section_set = set(sections) if sections is not None else set(ALLOWED_SECTIONS)
    final_record = _selective_merge(existing, reviewed_merged, section_set)
    changed = _changed_top_level_fields(existing, final_record)

    db.replace_paper_record(
        paper_id=paper_id,
        record=final_record,
        source_path=str(paper_markdown_path),
    )

    return ReviewUpdateResult(
        paper_id=paper_id,
        sections_updated=sorted(section_set),
        changed_top_level_fields=changed,
        confidence_before=existing.extraction_confidence,
        confidence_after=final_record.extraction_confidence,
    )
