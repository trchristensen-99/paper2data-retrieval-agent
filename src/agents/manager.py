from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from agents import Agent

from src.agents.data_availability import (
    DataAvailabilityOutput,
    data_availability_agent,
    run_data_availability_agent,
)
from src.agents.metadata import metadata_agent, run_metadata_agent
from src.agents.metadata_enrichment import (
    metadata_enrichment_agent,
    run_metadata_enrichment_agent,
)
from src.agents.methods import methods_agent, run_methods_agent
from src.agents.quality_control import quality_control_agent, run_quality_control_agent
from src.agents.results import results_agent, run_results_agent
from src.agents.synthesis import fallback_synthesis, run_synthesis_agent, synthesis_agent
from src.schemas.models import DataAvailabilityReport, PaperRecord, SynthesisInput, SynthesisOutput
from src.utils.config import MODELS
from src.utils.confidence import compute_extraction_confidence
from src.utils.logging import log_event, reset_events
from src.utils.taxonomy import normalize_category_subcategory


manager_agent = Agent(
    name="manager_agent",
    model=MODELS.manager,
    instructions=(
        "You orchestrate handoffs across metadata -> methods -> results -> "
        "data_availability -> synthesis."
    ),
    handoffs=[
        metadata_agent,
        methods_agent,
        results_agent,
        data_availability_agent,
        quality_control_agent,
        metadata_enrichment_agent,
        synthesis_agent,
    ],
)


def _print_step_timing(step: str, seconds: float) -> None:
    print(f"[pipeline] step={step} duration_seconds={seconds:.2f}", flush=True)


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _extract_year_hint(text: str) -> str | None:
    patterns = [
        r"Published:\s*([0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?)",
        r"Accepted:\s*([0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?)",
        r"Received:\s*([0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    years = re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b", text)
    return years[-1] if years else None


def _derive_keywords(
    metadata_keywords: list[str],
    category: str | None,
    subcategory: str | None,
    assay_types: list[str],
    organisms: list[str],
    journal: str | None,
) -> list[str]:
    raw: list[str] = []
    raw.extend(metadata_keywords)
    raw.extend([category or "", subcategory or ""])
    raw.extend(assay_types[:4])
    raw.extend(organisms[:3])
    raw.append(journal or "")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = item.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out[:8]


def _missing_key_metadata_fields(
    *,
    title: str,
    authors: list[str],
    journal: str | None,
    publication_date: str | None,
    keywords: list[str],
) -> list[str]:
    missing: list[str] = []
    if not title.strip():
        missing.append("metadata.title")
    if not authors:
        missing.append("metadata.authors")
    if _is_blank(journal):
        missing.append("metadata.journal")
    if _is_blank(publication_date):
        missing.append("metadata.publication_date")
    if not keywords:
        missing.append("metadata.keywords")
    return missing


@dataclass
class PipelineArtifacts:
    record: PaperRecord
    retrieval_report_markdown: str
    retrieval_log_markdown: str
    step_timings_seconds: dict[str, float]
    pipeline_duration_seconds: float
    quality_notes: str | None = None


async def run_pipeline(paper_markdown: str) -> PipelineArtifacts:
    pipeline_start = perf_counter()
    step_timings_seconds: dict[str, float] = {}

    reset_events()
    log_event("pipeline.start", {"chars": len(paper_markdown)})

    step_start = perf_counter()
    metadata = await run_metadata_agent(paper_markdown)
    metadata.category, metadata.subcategory = normalize_category_subcategory(
        metadata.category, metadata.subcategory
    )
    step_timings_seconds["metadata"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "metadata", "seconds": step_timings_seconds["metadata"]})
    _print_step_timing("metadata", step_timings_seconds["metadata"])

    step_start = perf_counter()
    methods = await run_methods_agent(paper_markdown)
    step_timings_seconds["methods"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "methods", "seconds": step_timings_seconds["methods"]})
    _print_step_timing("methods", step_timings_seconds["methods"])

    step_start = perf_counter()
    results = await run_results_agent(paper_markdown)
    step_timings_seconds["results"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "results", "seconds": step_timings_seconds["results"]})
    _print_step_timing("results", step_timings_seconds["results"])

    step_start = perf_counter()
    try:
        data_availability: DataAvailabilityOutput = await run_data_availability_agent(paper_markdown)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.data_availability.error", {"error": str(exc)})
        data_availability = DataAvailabilityOutput(
            data_accessions=[],
            data_availability=DataAvailabilityReport(
                overall_status="not_checked",
                claimed_repositories=[],
                verified_repositories=[],
                discrepancies=[f"data_availability_agent_error: {exc}"],
                notes="Data availability agent failed; fallback used.",
            ),
        )
    step_timings_seconds["data_availability"] = perf_counter() - step_start
    log_event(
        "pipeline.step_timing",
        {"step": "data_availability", "seconds": step_timings_seconds["data_availability"]},
    )
    _print_step_timing("data_availability", step_timings_seconds["data_availability"])

    step_start = perf_counter()
    quality_check = await run_quality_control_agent(
        paper_markdown=paper_markdown,
        metadata=metadata,
        methods=methods,
        results=results,
        data_availability=data_availability.data_availability,
    )
    step_timings_seconds["quality_control"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "quality_control", "seconds": step_timings_seconds["quality_control"]})
    _print_step_timing("quality_control", step_timings_seconds["quality_control"])

    if quality_check.should_retry and quality_check.retry_instructions:
        log_event(
            "pipeline.quality_retry.start",
            {
                "missing_fields": quality_check.missing_fields,
                "suspicious_empty_fields": quality_check.suspicious_empty_fields,
                "retry_count": len(quality_check.retry_instructions),
            },
        )
        for instruction in quality_check.retry_instructions:
            agent_name = instruction.agent_name.strip().lower()
            reason = instruction.reason
            if agent_name == "metadata":
                metadata = await run_metadata_agent(paper_markdown, guidance=reason)
                metadata.category, metadata.subcategory = normalize_category_subcategory(
                    metadata.category, metadata.subcategory
                )
            elif agent_name == "methods":
                methods = await run_methods_agent(paper_markdown, guidance=reason)
            elif agent_name == "results":
                results = await run_results_agent(paper_markdown, guidance=reason)
            elif agent_name in {"data_availability", "data-availability"}:
                data_availability = await run_data_availability_agent(paper_markdown, guidance=reason)
            log_event("pipeline.quality_retry.step", {"agent": agent_name, "reason": reason})
        log_event("pipeline.quality_retry.end", {"notes": quality_check.notes})

    needs_enrichment = (
        (not metadata.doi)
        or (not metadata.pmid)
        or (not metadata.journal)
        or (metadata.journal.strip().lower() == "scientific data")
    )
    if needs_enrichment:
        step_start = perf_counter()
        enrichment = await run_metadata_enrichment_agent(metadata, paper_markdown)
        step_timings_seconds["metadata_enrichment"] = perf_counter() - step_start
        log_event(
            "pipeline.step_timing",
            {"step": "metadata_enrichment", "seconds": step_timings_seconds["metadata_enrichment"]},
        )
        _print_step_timing("metadata_enrichment", step_timings_seconds["metadata_enrichment"])

        if not metadata.doi and enrichment.doi:
            metadata.doi = enrichment.doi
        if not metadata.pmid and enrichment.pmid:
            metadata.pmid = enrichment.pmid
        if enrichment.journal and (
            (not metadata.journal)
            or (metadata.journal.strip().lower() == "scientific data")
            or (
                metadata.doi
                and enrichment.doi
                and metadata.doi.strip().lower() == enrichment.doi.strip().lower()
            )
        ):
            metadata.journal = enrichment.journal
        if not metadata.publication_date and enrichment.publication_date:
            metadata.publication_date = enrichment.publication_date
        log_event("pipeline.metadata_enrichment.applied", {"notes": enrichment.notes})

    missing_before_repair = _missing_key_metadata_fields(
        title=metadata.title,
        authors=metadata.authors,
        journal=metadata.journal,
        publication_date=metadata.publication_date,
        keywords=metadata.keywords,
    )
    if missing_before_repair:
        guidance = (
            "Repair metadata completeness. Fill these fields with best available evidence from paper text "
            f"and tools: {', '.join(missing_before_repair)}. "
            "Prefer precise values; do not leave these as null/empty."
        )
        repaired = await run_metadata_agent(paper_markdown, guidance=guidance)
        repaired.category, repaired.subcategory = normalize_category_subcategory(
            repaired.category, repaired.subcategory
        )
        if not metadata.authors and repaired.authors:
            metadata.authors = repaired.authors
        if _is_blank(metadata.journal) and not _is_blank(repaired.journal):
            metadata.journal = repaired.journal
        if _is_blank(metadata.publication_date) and not _is_blank(repaired.publication_date):
            metadata.publication_date = repaired.publication_date
        if not metadata.keywords and repaired.keywords:
            metadata.keywords = repaired.keywords
        log_event(
            "pipeline.metadata_repair",
            {
                "missing_before_repair": missing_before_repair,
                "title": metadata.title,
            },
        )

    # Final deterministic guard so these key fields are never null/empty.
    if not metadata.title.strip():
        metadata.title = "Unknown title"
    if not metadata.authors:
        metadata.authors = ["Unknown author"]
    if _is_blank(metadata.journal):
        metadata.journal = "Unknown venue"
    if _is_blank(metadata.publication_date):
        metadata.publication_date = _extract_year_hint(paper_markdown) or "Unknown"
    if not metadata.keywords:
        metadata.keywords = _derive_keywords(
            metadata.keywords,
            metadata.category,
            metadata.subcategory,
            methods.assay_types,
            methods.organisms,
            metadata.journal,
        ) or ["unspecified"]

    synthesis_input = SynthesisInput(
        metadata=metadata,
        methods=methods,
        results=results,
        data_accessions=data_availability.data_accessions,
        data_availability=data_availability.data_availability,
    )

    step_start = perf_counter()
    try:
        synthesis: SynthesisOutput = await run_synthesis_agent(synthesis_input)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.synthesis.error", {"error": str(exc)})
        synthesis = fallback_synthesis(synthesis_input)
    step_timings_seconds["synthesis"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "synthesis", "seconds": step_timings_seconds["synthesis"]})
    _print_step_timing("synthesis", step_timings_seconds["synthesis"])

    confidence = compute_extraction_confidence(
        metadata=metadata,
        methods=methods,
        results=results,
        data_availability=data_availability.data_availability,
        quality_check=quality_check,
    )
    synthesis.record.extraction_confidence = confidence.score
    log_event("pipeline.confidence", confidence.as_dict())

    pipeline_duration_seconds = perf_counter() - pipeline_start
    log_event(
        "pipeline.end",
        {
            "confidence": synthesis.record.extraction_confidence,
            "pipeline_duration_seconds": pipeline_duration_seconds,
        },
    )
    print(f"[pipeline] total_duration_seconds={pipeline_duration_seconds:.2f}", flush=True)
    return PipelineArtifacts(
        record=synthesis.record,
        retrieval_report_markdown=synthesis.retrieval_report_markdown,
        retrieval_log_markdown=synthesis.retrieval_log_markdown,
        step_timings_seconds=step_timings_seconds,
        pipeline_duration_seconds=pipeline_duration_seconds,
        quality_notes=quality_check.notes,
    )
