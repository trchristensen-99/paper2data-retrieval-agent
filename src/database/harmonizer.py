from __future__ import annotations

import json
import re
from typing import Any

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel, Field

from src.schemas.models import DataAccession, DataAvailabilityReport, PaperRecord
from src.utils.config import MODELS
from src.utils.retry import run_with_rate_limit_retry


class HarmonizationOutput(BaseModel):
    merged_record: PaperRecord
    summary: str
    conflicts_resolved: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


harmonizer_agent = Agent(
    name="record_harmonizer_agent",
    model=MODELS.harmonizer,
    instructions=(
        "Merge two PaperRecord objects for the same paper into one canonical record. "
        "Resolve semantic duplicates (e.g., H. sapiens vs Homo sapiens). "
        "Prefer more specific, evidence-backed values. "
        "When conflicting, keep the value most consistent with the paper-level metadata and methods context. "
        "Do not drop unique useful facts unless they are clearly contradictory or wrong."
    ),
    output_type=AgentOutputSchema(HarmonizationOutput, strict_json_schema=False),
)


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _uniq_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        k = _norm_text(v)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _merge_accessions(a: list[DataAccession], b: list[DataAccession]) -> list[DataAccession]:
    merged: dict[str, DataAccession] = {}
    for item in a + b:
        key = f"{item.repository.lower()}::{item.accession_id.lower()}"
        if key not in merged:
            merged[key] = item
            continue
        cur = merged[key]
        merged[key] = DataAccession(
            accession_id=cur.accession_id,
            category=cur.category if cur.category != "external_reference" else item.category,
            repository=cur.repository,
            url=cur.url or item.url,
            description=cur.description if len(cur.description) >= len(item.description) else item.description,
            data_format=cur.data_format or item.data_format,
            is_accessible=cur.is_accessible if cur.is_accessible is not None else item.is_accessible,
            file_count=cur.file_count if cur.file_count is not None else item.file_count,
            files_listed=cur.files_listed or item.files_listed,
            total_size_bytes=cur.total_size_bytes if cur.total_size_bytes is not None else item.total_size_bytes,
            estimated_download_seconds=(
                cur.estimated_download_seconds
                if cur.estimated_download_seconds is not None
                else item.estimated_download_seconds
            ),
            download_probe_url=cur.download_probe_url or item.download_probe_url,
        )
    return list(merged.values())


def fallback_harmonize(existing: PaperRecord, incoming: PaperRecord) -> HarmonizationOutput:
    merged = existing.model_copy(deep=True)

    # Metadata fields: prefer non-empty incoming values.
    for field in [
        "title",
        "doi",
        "pmid",
        "journal",
        "publication_date",
        "conflicts_of_interest",
        "paper_type",
        "license",
        "category",
        "subcategory",
    ]:
        cur = getattr(merged.metadata, field)
        new = getattr(incoming.metadata, field)
        if _is_empty(cur) and not _is_empty(new):
            setattr(merged.metadata, field, new)

    merged.metadata.authors = _uniq_list(merged.metadata.authors + incoming.metadata.authors)
    merged.metadata.keywords = _uniq_list(merged.metadata.keywords + incoming.metadata.keywords)
    merged.metadata.funding_sources = _uniq_list(
        merged.metadata.funding_sources + incoming.metadata.funding_sources
    )

    # Methods: keep richer text and union lists.
    merged.methods.organisms = _uniq_list(merged.methods.organisms + incoming.methods.organisms)
    merged.methods.cell_types = _uniq_list(merged.methods.cell_types + incoming.methods.cell_types)
    merged.methods.assay_types = _uniq_list(merged.methods.assay_types + incoming.methods.assay_types)
    merged.methods.statistical_tests = _uniq_list(
        merged.methods.statistical_tests + incoming.methods.statistical_tests
    )
    merged.methods.sample_sizes = {**merged.methods.sample_sizes, **incoming.methods.sample_sizes}
    if len(incoming.methods.experimental_design) > len(merged.methods.experimental_design):
        merged.methods.experimental_design = incoming.methods.experimental_design
    if len(incoming.methods.methods_completeness) > len(merged.methods.methods_completeness):
        merged.methods.methods_completeness = incoming.methods.methods_completeness

    # Results: merge findings by normalized claim+metric.
    existing_keys = {
        (_norm_text(f.claim), _norm_text(f.metric)) for f in merged.results.experimental_findings
    }
    for finding in incoming.results.experimental_findings:
        key = (_norm_text(finding.claim), _norm_text(finding.metric))
        if key not in existing_keys:
            merged.results.experimental_findings.append(finding)
            existing_keys.add(key)
    merged.results.synthesized_claims = _uniq_list(
        merged.results.synthesized_claims + incoming.results.synthesized_claims
    )
    if len(incoming.results.dataset_properties) > len(merged.results.dataset_properties):
        merged.results.dataset_properties = incoming.results.dataset_properties
    if len(incoming.results.method_benchmarks) > len(merged.results.method_benchmarks):
        merged.results.method_benchmarks = incoming.results.method_benchmarks
    if incoming.results.paper_type and not merged.results.paper_type:
        merged.results.paper_type = incoming.results.paper_type

    merged.data_accessions = _merge_accessions(merged.data_accessions, incoming.data_accessions)
    merged.code_repositories = _uniq_list(merged.code_repositories + incoming.code_repositories)

    merged.data_availability = DataAvailabilityReport(
        overall_status=merged.data_availability.overall_status
        if merged.data_availability.overall_status != "not_checked"
        else incoming.data_availability.overall_status,
        claimed_repositories=_uniq_list(
            merged.data_availability.claimed_repositories
            + incoming.data_availability.claimed_repositories
        ),
        verified_repositories=_uniq_list(
            merged.data_availability.verified_repositories
            + incoming.data_availability.verified_repositories
        ),
        discrepancies=_uniq_list(
            merged.data_availability.discrepancies + incoming.data_availability.discrepancies
        ),
        notes=merged.data_availability.notes
        if len(merged.data_availability.notes) >= len(incoming.data_availability.notes)
        else incoming.data_availability.notes,
    )

    merged.extraction_confidence = max(existing.extraction_confidence, incoming.extraction_confidence)
    merged.extraction_timestamp = incoming.extraction_timestamp

    return HarmonizationOutput(
        merged_record=merged,
        summary="Fallback merge applied using deterministic rules.",
        confidence=0.75,
        conflicts_resolved=["Merged metadata/methods/results/data-accession fields deterministically."],
        unresolved_conflicts=[],
    )


async def harmonize_records(existing: PaperRecord, incoming: PaperRecord) -> HarmonizationOutput:
    payload = {
        "existing_record": existing.model_dump(),
        "incoming_record": incoming.model_dump(),
        "rules": [
            "Keep semantically equivalent values once.",
            "Prefer specific and verifiable values.",
            "When uncertain, preserve both in lists and mention unresolved conflicts.",
        ],
    }
    try:
        result = await run_with_rate_limit_retry(
            lambda: Runner.run(harmonizer_agent, input=json.dumps(payload)),
            max_retries=2,
        )
        output = result.final_output
        if not isinstance(output, HarmonizationOutput):
            output = HarmonizationOutput.model_validate(output)
        return output
    except Exception:
        return fallback_harmonize(existing, incoming)
