from __future__ import annotations

import html
import os
import re
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from agents import Agent

from src.agents.archetype import ArchetypeOutput, archetype_agent, run_archetype_agent
from src.agents.anatomy import anatomy_agent, run_anatomy_agent
from src.agents.data_availability import (
    DataAvailabilityOutput,
    data_availability_agent,
    fallback_data_availability_from_text,
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
from src.schemas.models import (
    AssayTypeMapping,
    DataAsset,
    DataAccession,
    DataAvailabilityReport,
    DatasetColumn,
    DatasetProfile,
    DescriptiveStat,
    ExperimentalDesignStep,
    ExtractedTable,
    ExperimentalFinding,
    FindingsBlock,
    MethodTool,
    QuantitativeDatum,
    InterpretiveClaim,
    PrismaFlow,
    Provenance,
    PaperRecord,
    RelatedResource,
    SampleSizeRecord,
    SynthesisInput,
    SynthesisOutput,
)
from src.utils.config import MODELS
from src.utils.confidence import compute_extraction_confidence
from src.utils.logging import log_event, reset_events
from src.utils.taxonomy import is_valid_category_subcategory, normalize_category_subcategory


manager_agent = Agent(
    name="manager_agent",
    model=MODELS.manager,
    instructions=(
        "You orchestrate handoffs across metadata -> methods -> results -> "
        "data_availability -> synthesis."
    ),
    handoffs=[
        archetype_agent,
        anatomy_agent,
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


def _extract_publication_date(text: str) -> str | None:
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    pattern = re.compile(
        r"\b(?:published(?:\s+online)?|accepted|received)\s*[:\-]?\s*"
        r"([0-9]{1,2})\s+([A-Za-z]+)\s+([12][0-9]{3})\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        day = int(match.group(1))
        month_name = match.group(2).strip().lower()
        year = int(match.group(3))
        month = month_map.get(month_name)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"
    iso_match = re.search(
        r"\b(?:published(?:\s+online)?|accepted|received)\s*[:\-]?\s*([12][0-9]{3}-[0-9]{2}-[0-9]{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if iso_match:
        return iso_match.group(1)
    return None


def _normalize_publication_status(publication_date: str | None) -> str | None:
    if not publication_date:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", publication_date.strip()):
        return None
    try:
        dt = datetime.strptime(publication_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None
    now = datetime.now(timezone.utc)
    if dt > now:
        return "advance_access"
    return "confirmed"


def _normalize_license_to_spdx(value: str | None) -> str | None:
    if not value:
        return value
    text = re.sub(r"\s+", " ", value.strip().lower())
    if "creative commons attribution" in text or "cc by 4.0" in text or "cc-by-4.0" in text:
        return "CC-BY-4.0"
    if "cc0" in text or "public domain dedication" in text:
        return "CC0-1.0"
    return value.strip()


def _normalize_venue_name(journal: str | None) -> str | None:
    if journal is None:
        return None
    value = journal.strip()
    if not value:
        return None
    if value.lower() == "scientific data":
        return "Nature Scientific Data"
    return value


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


def _clean_input_text(raw: str) -> str:
    cleaned = html.unescape(raw)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _infer_paper_type(metadata_title: str, paper_markdown: str) -> str:
    text = f"{metadata_title}\n{paper_markdown}".lower()
    if any(k in text for k in ("dataset descriptor", "data descriptor", "dataset", "repository")):
        return "dataset_descriptor"
    if any(k in text for k in ("systematic review", "scoping review", "literature review")):
        return "review"
    if any(k in text for k in ("meta-analysis", "meta analysis")):
        return "meta_analysis"
    if any(k in text for k in ("we propose", "benchmark", "method", "pipeline")):
        return "methods"
    if any(k in text for k in ("commentary", "perspective", "opinion")):
        return "commentary"
    return "experimental"


def _missing_key_metadata_fields(
    *,
    title: str,
    authors: list[str],
    journal: str | None,
    publication_date: str | None,
    keywords: list[str],
    paper_type: str | None,
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
    if _is_blank(paper_type):
        missing.append("metadata.paper_type")
    return missing


def _anatomy_guidance_blob(sections: list[str], tables: list[str], figures: list[str], urls: list[str]) -> str:
    return (
        "Paper anatomy hints:\n"
        f"- Sections: {sections[:12]}\n"
        f"- Tables: {tables[:12]}\n"
        f"- Figures: {figures[:12]}\n"
        f"- URLs: {urls[:20]}\n"
    )


def _norm_url_ocr(url: str | None) -> str | None:
    if not url:
        return url
    fixed = url.strip()
    fixed = re.sub(r"https?://(?:www\.)?fgshare\.com", "https://figshare.com", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\b10\.6084/m9\.fgshare\.", "10.6084/m9.figshare.", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"([?&])fle=([0-9]+)", r"\1file=\2", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"([?&])file=([0-9]{8})[0-9]+", r"\1file=\2", fixed, flags=re.IGNORECASE)
    return fixed


def _apply_url_canonicalization(
    *,
    data_accessions: list[DataAccession],
    related_resources: list[RelatedResource],
) -> None:
    for accession in data_accessions:
        accession.url = _norm_url_ocr(accession.url)
        accession.accession_id = _norm_url_ocr(accession.accession_id) or accession.accession_id
    for resource in related_resources:
        resource.url = _norm_url_ocr(resource.url)


def _extract_first_int(pattern: str, text: str, flags: int = re.IGNORECASE) -> int | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    return int(re.sub(r"[^0-9]", "", match.group(1)))


def _locate_provenance(text: str, needle: str, *, source_table: str | None = None) -> Provenance | None:
    token = (needle or "").strip()
    if not token:
        return None
    token_low = token.lower()
    lines = text.splitlines()
    lines_with_ends = text.splitlines(keepends=True)
    page_markers = []
    for idx, line in enumerate(lines, start=1):
        m = re.search(r"\bpage\s+([0-9]{1,4})\b", line, flags=re.IGNORECASE)
        if m:
            page_markers.append((idx, int(m.group(1))))
    for idx, line in enumerate(lines, start=1):
        if token_low in line.lower():
            source_page: int | None = None
            if "\f" in text:
                prefix = "".join(lines_with_ends[: idx - 1])
                source_page = prefix.count("\f") + 1
            if source_page is None and page_markers:
                for marker_line, marker_page in page_markers:
                    if marker_line <= idx:
                        source_page = marker_page
                    else:
                        break
            return Provenance(
                source_page=source_page,
                source_section=None,
                source_table=source_table,
                line_start=idx,
                line_end=idx,
                text_segment=line.strip()[:300],
            )
    return None


def _normalize_accession_identifiers(accessions: list[DataAccession]) -> None:
    for accession in accessions:
        raw_id = (accession.accession_id or "").strip()
        raw_url = (accession.url or "").strip()
        joined = f"{raw_id} {raw_url}".lower()
        system = "OTHER"
        normalized_id = raw_id
        if m := re.search(r"(10\.\d{4,9}/[^\s]+)", joined, flags=re.IGNORECASE):
            doi = m.group(1).rstrip(".,;")
            system = "DOI"
            normalized_id = f"doi:{doi.lower()}"
        elif re.search(r"\bGSE\d+\b", raw_id, flags=re.IGNORECASE):
            token = re.search(r"\bGSE\d+\b", raw_id, flags=re.IGNORECASE).group(0).upper()
            system = "GEO"
            normalized_id = f"geo:{token}"
        elif re.search(r"\b(?:SRP|SRR|SRA)\d+\b", raw_id, flags=re.IGNORECASE):
            token = re.search(r"\b(?:SRP|SRR|SRA)\d+\b", raw_id, flags=re.IGNORECASE).group(0).upper()
            system = "SRA"
            normalized_id = f"sra:{token}"
        elif raw_id:
            normalized_id = f"id:{raw_id.lower()}"

        repo_name = (accession.repository or "").strip().lower()
        if any(k in repo_name for k in ("figshare", "zenodo", "dryad", "dataverse", "osf")):
            repo_type = "generalist"
        elif any(k in repo_name for k in ("geo", "sra", "ena", "arrayexpress")):
            repo_type = "domain_specific"
        else:
            repo_type = "other"

        accession.system = system
        accession.normalized_id = normalized_id
        accession.repository_type = repo_type


def _infer_domain_profile(category: str | None, paper_type: str | None, paper_markdown: str) -> str:
    c = (category or "").strip().lower()
    p = (paper_type or "").strip().lower()
    text = paper_markdown.lower()
    if c in {"biology", "medicine_health", "environmental_science", "climate_science"}:
        return "bio"
    if p in {"dataset_descriptor", "review", "meta_analysis"}:
        return "meta_research"
    if c in {"data_science_ai", "computer_science", "mathematics_statistics"}:
        return "computational"
    if any(k in text for k in ("rna-seq", "chip-seq", "single-cell", "mouse", "homo sapiens")):
        return "bio"
    return "general"


def _assay_mapping_for(raw_value: str) -> AssayTypeMapping:
    raw = re.sub(r"\s+", " ", raw_value.strip())
    low = raw.lower()
    mapping = [
        ("rna-seq", "RNA sequencing assay", "OBI:0001271", "OBI"),
        ("chip-seq", "chromatin immunoprecipitation sequencing", "OBI:0000716", "OBI"),
        ("atac-seq", "ATAC-seq assay", "OBI:0002039", "OBI"),
        ("mpra", "massively parallel reporter assay", "OBI:0002675", "OBI"),
        ("manual literature curation", "literature curation", "NCIT:C159670", "NCIT"),
        ("scoping review", "scoping review", "NCIT:C17649", "NCIT"),
        ("systematic review", "systematic review", "NCIT:C40514", "NCIT"),
        ("meta-analysis", "meta-analysis", "NCIT:C53326", "NCIT"),
        ("statistical", "statistical analysis", "NCIT:C25656", "NCIT"),
        ("benchmark", "benchmarking assessment", "NCIT:C142678", "NCIT"),
    ]
    for needle, mapped, ont, vocab in mapping:
        if needle in low:
            return AssayTypeMapping(raw=raw, mapped_term=mapped, ontology_id=ont, vocabulary=vocab)
    mapped = re.sub(r"[^a-z0-9]+", "_", low).strip("_").upper() or "OTHER_ANALYSIS"
    return AssayTypeMapping(raw=raw, mapped_term=mapped, ontology_id=None, vocabulary=None)


def _extract_experimental_design_steps(
    paper_markdown: str,
    methods_summary: str,
    prisma_flow: PrismaFlow | None,
) -> list[ExperimentalDesignStep]:
    steps: list[ExperimentalDesignStep] = []
    flow = prisma_flow.as_compact_dict() if isinstance(prisma_flow, PrismaFlow) else {}
    db_total = flow.get("database_records_total")
    if db_total:
        step = ExperimentalDesignStep(
            step=1,
            action="Database search",
            tools=[
                MethodTool(name="ACM Digital Library", software_type="literature_database"),
                MethodTool(name="IEEE Xplore", software_type="literature_database"),
                MethodTool(name="Web of Science", software_type="literature_database"),
            ],
            count=int(db_total),
            context="Initial identification from indexed databases.",
            provenance=_locate_provenance(paper_markdown, "records from", source_table=None),
        )
        steps.append(step)
    screened = flow.get("screened") or flow.get("records_after_duplicate_removal")
    if screened:
        steps.append(
            ExperimentalDesignStep(
                step=len(steps) + 1,
                action="Screening",
                tools=[MethodTool(name="Covidence", software_type="screening_platform")]
                if "covidence" in paper_markdown.lower()
                else [],
                criteria="Title/Abstract" if "title" in paper_markdown.lower() and "abstract" in paper_markdown.lower() else None,
                count=int(screened),
                excluded=flow.get("excluded_title_abstract"),
                context="Primary screening phase.",
                provenance=_locate_provenance(paper_markdown, "screened", source_table=None),
            )
        )
    full_text = flow.get("full_text_review")
    if full_text:
        steps.append(
            ExperimentalDesignStep(
                step=len(steps) + 1,
                action="Eligibility review",
                criteria="Full text",
                count=int(full_text),
                excluded=flow.get("excluded_full_text"),
                context="Full-text eligibility assessment.",
                provenance=_locate_provenance(paper_markdown, "full-text", source_table=None),
            )
        )
    included = flow.get("included")
    if included:
        steps.append(
            ExperimentalDesignStep(
                step=len(steps) + 1,
                action="Inclusion",
                included=int(included),
                tools=[
                    MethodTool(
                        name="DisGeNET RDF",
                        version="modified" if "modified disgenet" in paper_markdown.lower() else None,
                        citation="[cite: 1]" if "disgenet" in paper_markdown.lower() else None,
                        software_type="semantic_model",
                    )
                ]
                if "disgenet" in paper_markdown.lower()
                else [],
                context="Final included studies/data sources.",
                provenance=_locate_provenance(paper_markdown, "included", source_table=None),
            )
        )
    if not steps and methods_summary.strip():
        steps.append(
            ExperimentalDesignStep(
                step=1,
                action="Method summary",
                context=methods_summary[:280],
                provenance=_locate_provenance(paper_markdown, methods_summary[:40], source_table=None),
            )
        )
    return steps


def _extract_quantitative_findings_from_text(
    paper_markdown: str,
    existing: list[ExperimentalFinding],
) -> list[ExperimentalFinding]:
    if existing:
        return existing
    text = paper_markdown
    patterns = [
        (r"fairness\s*\(?\s*([0-9]+(?:\.[0-9]+)?)%\s*\)?", "Principle share: fairness", "principle_share"),
        (r"transparency\s*\(?\s*([0-9]+(?:\.[0-9]+)?)%\s*\)?", "Principle share: transparency", "principle_share"),
        (r"privacy\s*\(?\s*([0-9]+(?:\.[0-9]+)?)%\s*\)?", "Principle share: privacy", "principle_share"),
        (r"trust\s*\(?\s*([0-9]+(?:\.[0-9]+)?)%\s*\)?", "Principle share: trust", "principle_share"),
    ]
    out: list[ExperimentalFinding] = []
    for pattern, claim, metric in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        value = m.group(1)
        prov = _locate_provenance(text, m.group(0))
        out.append(
            ExperimentalFinding(
                claim=claim,
                metric=metric,
                value=value,
                unit="percent",
                statistical_significance=None,
                effect_size=None,
                confidence_interval=None,
                comparison=None,
                context="Distributional summary reported in paper text/figures.",
                confidence=0.82,
                provenance=prov,
            )
        )
    return out


def _derive_findings_from_dataset_properties(
    dataset_properties: list[DescriptiveStat],
) -> list[ExperimentalFinding]:
    out: list[ExperimentalFinding] = []
    for prop in dataset_properties:
        if not re.search(r"[0-9]", prop.value):
            continue
        out.append(
            ExperimentalFinding(
                claim=f"{prop.property} = {prop.value}",
                metric=prop.variable or prop.property,
                value=prop.value,
                unit=prop.unit,
                statistical_significance=prop.statistical_significance,
                effect_size=None,
                confidence_interval=None,
                comparison=None,
                context=prop.context,
                confidence=max(0.6, min(0.95, prop.confidence)),
                provenance=prop.provenance,
            )
        )
    return out[:25]


def _to_number(value: str | None) -> int | float | None:
    if value is None:
        return None
    txt = str(value).strip().replace(",", "")
    if not txt:
        return None
    try:
        if "." in txt:
            return float(txt)
        return int(txt)
    except Exception:  # noqa: BLE001
        return None


def _standardize_metric_key(label: str) -> tuple[str, str]:
    low = label.strip().lower()
    mappings = [
        ("association", "n_associations_total", "dataset_size"),
        ("record_count", "n_records_total", "dataset_size"),
        ("row", "n_rows_total", "dataset_size"),
        ("column", "n_columns_total", "dataset_size"),
        ("paper", "n_papers_total", "provenance"),
        ("publication", "n_publications_total", "provenance"),
        ("pmid", "n_publications_provenance", "provenance"),
        ("gene", "n_genes_total", "entity_count"),
        ("disease", "n_diseases_total", "entity_count"),
        ("sample", "n_samples_total", "sample_size"),
    ]
    for needle, std_key, category in mappings:
        if needle in low:
            return std_key, category
    token = re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "n_metric_total"
    return token[:80], "other"


def _derive_sample_size_records(methods, paper_markdown: str) -> None:
    records: list[SampleSizeRecord] = []
    seen: set[str] = set()
    for key, raw_value in (methods.sample_sizes or {}).items():
        label = str(key)
        value_obj = raw_value
        if isinstance(raw_value, dict) and "value" in raw_value:
            value_obj = raw_value["value"]
        value = _to_number(str(value_obj))
        if value is None:
            value = str(value_obj)
        std_key, category = _standardize_metric_key(label)
        sig = f"{std_key}::{value}"
        if sig in seen:
            continue
        seen.add(sig)
        missingness_status = "reported"
        if isinstance(value, str) and not value.strip():
            missingness_status = "not_reported"
        records.append(
            SampleSizeRecord(
                standardized_key=std_key,
                original_label=label,
                value=value,
                unit="count" if isinstance(value, (int, float)) else None,
                category=category,
                missingness_status=missingness_status,
                provenance=_locate_provenance(paper_markdown, label),
            )
        )
    methods.sample_size_records = records


def _set_missingness(methods, paper_type: str | None) -> None:
    status: dict[str, str] = {}
    pt = (paper_type or "").strip().lower()
    status["sample_size"] = "reported" if methods.sample_size_records else ("not_applicable" if pt in {"commentary"} else "not_reported")
    status["statistical_tests"] = "reported" if methods.statistical_tests else ("not_applicable" if pt in {"dataset_descriptor", "commentary"} else "not_reported")
    status["organisms"] = "reported" if methods.organisms else ("not_applicable" if pt in {"software_benchmark", "commentary"} else "not_reported")
    status["assay_types"] = "reported" if methods.assay_types else "not_reported"
    methods.missingness = status


def _link_tool_entities(methods) -> None:
    known_ids = {
        "cytoscape": "Wikidata:Q1149474",
        "covidence": "Wikidata:Q130427040",
        "web of science": "Wikidata:Q11921",
        "omim": "Wikidata:Q7089225",
        "disgenet rdf": "Wikidata:Q122953504",
        "jupyter": "Wikidata:Q5561638",
        "ieee xplore": "Wikidata:Q5345586",
        "acm digital library": "Wikidata:Q4651526",
    }
    for step in methods.experimental_design_steps:
        fixed_tools: list[MethodTool] = []
        for tool in step.tools:
            name = (tool.name or "").strip()
            low = name.lower()
            entity_id = tool.entity_id
            if not entity_id:
                for k, v in known_ids.items():
                    if k in low:
                        entity_id = v
                        break
            fixed_tools.append(
                MethodTool(
                    name=name or "Unknown tool",
                    entity_id=entity_id,
                    version=tool.version,
                    citation=tool.citation,
                    software_type=tool.software_type,
                )
            )
        step.tools = fixed_tools


def _build_findings_block(results) -> None:
    q_data: list[QuantitativeDatum] = []
    for idx, finding in enumerate(results.experimental_findings):
        q_data.append(
            QuantitativeDatum(
                entity=None,
                measurement=finding.metric,
                value=finding.value,
                unit=finding.unit,
                p_value=finding.p_value,
                context=finding.context,
                linked_finding_index=idx,
                provenance=finding.provenance,
            )
        )
    interpretive: list[InterpretiveClaim] = []
    for claim in results.synthesized_claims:
        claim_low = claim.lower()
        support = "supported_by_data" if q_data and any(t in claim_low for t in ("increase", "decrease", "associated", "significant", "suggest")) else "unsupported_or_unclear"
        interpretive.append(
            InterpretiveClaim(
                claim=claim,
                support_status=support,
                linked_data_id=0 if q_data else None,
                provenance=None,
            )
        )
    results.findings = FindingsBlock(quantitative_data=q_data, interpretive_claims=interpretive)


def _export_large_tables(tables: list[ExtractedTable], *, row_threshold: int = 20) -> None:
    export_root = os.path.join("outputs", "table_exports")
    os.makedirs(export_root, exist_ok=True)
    for table in tables:
        rows = table.data or []
        if len(rows) <= row_threshold:
            continue
        columns = table.columns or sorted({k for row in rows for k in row.keys()})
        safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", table.table_id or "table")
        digest = hashlib.sha1((table.table_id + str(len(rows))).encode("utf-8")).hexdigest()[:10]
        out_path = os.path.join(export_root, f"{safe_id}_{digest}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        table.storage_path = out_path
        table.summary = (table.summary or "") + f" Exported {len(rows)} rows to {out_path}."
        table.data = []


def _normalize_table_topology(tables: list[ExtractedTable]) -> list[ExtractedTable]:
    if not tables:
        return []
    grouped: dict[str, list[ExtractedTable]] = {}
    for t in tables:
        base = re.sub(r"[-_ ]?part[ab]\b.*$", "", (t.table_id or "").strip(), flags=re.IGNORECASE).strip()
        key = base.lower() or (t.table_id or "").lower()
        grouped.setdefault(key, []).append(t)

    out: list[ExtractedTable] = []
    for _, group in grouped.items():
        has_parts = any(re.search(r"part[ab]", (t.table_id or ""), flags=re.IGNORECASE) for t in group)
        for t in group:
            if has_parts and re.search(r"^table\s*\d+\s*$", (t.table_id or "").strip(), flags=re.IGNORECASE):
                continue
            if not t.data:
                out.append(t)
                continue
            # Convert section-header rows into explicit category column.
            columns = list(t.columns or [])
            if not columns:
                first_row = t.data[0]
                columns = list(first_row.keys())
            primary = columns[0] if columns else None
            value_cols = columns[1:] if len(columns) > 1 else []
            current_category: str | None = None
            normalized_rows: list[dict[str, str]] = []
            need_category = False
            for row in t.data:
                if not isinstance(row, dict):
                    continue
                row_primary = str(row.get(primary, "")).strip() if primary else ""
                non_primary_values = [str(row.get(c, "")).strip() for c in value_cols]
                numeric_like = [v for v in non_primary_values if re.search(r"\d", v)]
                blank_values = all(not v for v in non_primary_values)
                if row_primary and not numeric_like and blank_values and value_cols:
                    current_category = row_primary
                    need_category = True
                    continue
                new_row = {str(k): str(v) for k, v in row.items()}
                if current_category:
                    new_row["category"] = current_category
                    need_category = True
                normalized_rows.append(new_row)
            if need_category:
                t.data = normalized_rows
                if "category" not in columns:
                    t.columns = ["category"] + columns
            else:
                t.data = normalized_rows
            out.append(t)
    # Deduplicate by table_id + columns.
    deduped: list[ExtractedTable] = []
    seen: set[str] = set()
    for t in out:
        key = f"{(t.table_id or '').lower()}::{','.join((t.columns or []))}::{len(t.data)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def _reclassify_dataset_counts_to_sample_sizes(methods, results) -> None:
    moved_prefixes = (
        "record_count",
        "data_point_count",
        "source_corpus_size",
        "unique_",
        "total_",
        "count_",
        "gene_disease",
        "causative_genes",
        "diseases",
    )
    keep: list[DescriptiveStat] = []
    for prop in results.dataset_properties:
        p = (prop.property or "").strip().lower()
        if any(p.startswith(prefix) for prefix in moved_prefixes):
            methods.sample_sizes[p] = prop.value
        else:
            keep.append(prop)
    results.dataset_properties = keep


def _infer_asset_content_type(file_name: str) -> str:
    low = file_name.lower()
    if ("gene" in low and "disease" in low) or ("gene-rd" in low) or ("provenance" in low and "gene" in low):
        return "gene_disease_associations"
    if "readme" in low:
        return "metadata_readme"
    if low.endswith(".ipynb") or low.endswith(".py"):
        return "analysis_code"
    if "visualization" in low or "sunburst" in low:
        return "visualization_link"
    if low.endswith(".xlsx") or low.endswith(".csv") or low.endswith(".tsv"):
        return "tabular_dataset"
    return "dataset_asset"


def _derive_data_assets(
    *,
    data_accessions: list[DataAccession],
    dataset_profile: DatasetProfile | None,
    paper_markdown: str,
) -> list[DataAsset]:
    assets: list[DataAsset] = []
    seen: set[str] = set()
    row_hint = None
    if dataset_profile and dataset_profile.record_count is not None:
        row_hint = dataset_profile.record_count
    for accession in data_accessions:
        files = list(accession.files_listed or [])
        if not files and accession.url:
            maybe_name = accession.url.rstrip("/").split("/")[-1]
            if "." in maybe_name:
                files = [maybe_name]
        for file_name in files:
            fname = str(file_name).strip()
            if not fname:
                continue
            key = f"{(accession.accession_id or '').lower()}::{fname.lower()}"
            if key in seen:
                continue
            seen.add(key)
            assets.append(
                DataAsset(
                    content_type=_infer_asset_content_type(fname),
                    file_name=fname,
                    row_count=row_hint if _infer_asset_content_type(fname) in {"tabular_dataset", "gene_disease_associations"} else None,
                    url=accession.download_probe_url or accession.url,
                    source_accession_id=accession.accession_id,
                    confidence=0.86 if accession.is_accessible else 0.72,
                    provenance=_locate_provenance(paper_markdown, fname),
                )
            )
    return assets


def _extract_prisma_flow(text: str, existing: dict[str, int] | None = None) -> dict[str, int]:
    flow: dict[str, int] = dict(existing or {})
    alias_map = {
        "records_identified": "database_records_total",
        "records_screened": "screened",
        "full_text_reviews": "full_text_review",
        "studies_included": "included",
    }
    for old_key, new_key in alias_map.items():
        if old_key in flow and new_key not in flow:
            value = flow.get(old_key)
            if isinstance(value, int):
                flow[new_key] = value
    patterns = {
        "acm_records": r"\b([0-9][0-9,]*)\s+(?:records?\s+)?from\s+acm(?:\s+digital\s+library|\s+dl)?\b",
        "ieee_records": r"\b([0-9][0-9,]*)\s+(?:records?\s+)?from\s+ieee\b",
        "citation_review_records": r"\b([0-9][0-9,]*)\s+(?:from\s+)?citation(?:\s+review)?\b",
        "expert_records": r"\b([0-9][0-9,]*)\s+(?:from\s+)?expert(?:\s+consultation)?\b",
        "duplicates_removed": r"\b([0-9][0-9,]*)\s+duplicates?\s+removed\b",
        "screened": r"\b([0-9][0-9,]*)\s+(?:records?\s+)?(?:were\s+)?screened\b",
        "excluded_title_abstract": r"\b([0-9][0-9,]*)\s+(?:were\s+)?excluded\s+(?:at|during)\s+title(?:\s*(?:and|&|/)\s*abstract)?\b",
        "full_text_review": r"\b([0-9][0-9,]*)\s+(?:full[\-\s]?text(?:\s+articles?)?)\s+(?:were\s+)?(?:assessed|reviewed)\b",
        "excluded_full_text": r"\b([0-9][0-9,]*)\s+(?:were\s+)?excluded\s+(?:at|during)\s+full[\-\s]?text\b",
        "included": r"\b([0-9][0-9,]*)\s+(?:papers?|studies|articles?|records?)\s+(?:were\s+)?included\b",
    }
    lowered = text.lower()
    for key, pattern in patterns.items():
        value = _extract_first_int(pattern, lowered)
        if value is not None:
            flow[key] = value
    title_abs_alt = _extract_first_int(r"title(?:\s*(?:and|&|/)\s*abstract)[^\n]{0,120}?\(n\s*=\s*([0-9][0-9,]*)\)", lowered)
    if title_abs_alt is not None:
        flow["excluded_title_abstract"] = title_abs_alt
    full_text_alt = _extract_first_int(r"full[\-\s]?text[^\n]{0,120}?\(n\s*=\s*([0-9][0-9,]*)\)[^\n]{0,80}excluded", lowered)
    if full_text_alt is not None:
        flow["excluded_full_text"] = full_text_alt
    if "database_records_total" not in flow and ("acm_records" in flow or "ieee_records" in flow):
        flow["database_records_total"] = int(flow.get("acm_records", 0)) + int(flow.get("ieee_records", 0))
    if "records_identified_total" not in flow or int(flow.get("records_identified_total", 0) or 0) <= 0:
        flow["records_identified_total"] = (
            int(flow.get("database_records_total", 0))
            + int(flow.get("citation_review_records", 0))
            + int(flow.get("expert_records", 0))
        )
    if "records_after_duplicate_removal" not in flow and "screened" in flow:
        flow["records_after_duplicate_removal"] = int(flow["screened"])
    if "excluded_title_abstract" not in flow and "screened" in flow and "full_text_review" in flow:
        diff = int(flow["screened"]) - int(flow["full_text_review"])
        if diff > 0:
            flow["excluded_title_abstract"] = diff
    if "excluded_full_text" not in flow and "full_text_review" in flow and "included" in flow:
        diff = int(flow["full_text_review"]) - int(flow["included"])
        if diff > 0:
            flow["excluded_full_text"] = diff
    for key in ("excluded_title_abstract", "excluded_full_text", "screened", "full_text_review", "included", "database_records_total"):
        if int(flow.get(key, 0) or 0) < 50 and key in {"excluded_title_abstract", "excluded_full_text", "screened", "full_text_review", "included", "database_records_total"}:
            flow.pop(key, None)
    # Remove likely year artifacts accidentally parsed as counts.
    for key in ("citation_review_records", "expert_records", "database_records_total", "records_identified_total"):
        value = int(flow.get(key, 0) or 0)
        if 1900 <= value <= 2100:
            flow.pop(key, None)
    return flow


def _prisma_model(flow: dict[str, int] | PrismaFlow | None) -> PrismaFlow:
    if isinstance(flow, PrismaFlow):
        model = flow
    elif isinstance(flow, dict):
        model = PrismaFlow.model_validate(flow)
    else:
        model = PrismaFlow()
    data = model.as_compact_dict()
    if data.get("records_identified_total") is None:
        total = int(data.get("database_records_total", 0)) + int(data.get("citation_review_records", 0)) + int(data.get("expert_records", 0))
        if total > 0:
            data["records_identified_total"] = total
    if data.get("records_after_duplicate_removal") is None and data.get("screened") is not None:
        data["records_after_duplicate_removal"] = int(data["screened"])
    return PrismaFlow.model_validate(data)


def _extract_figshare_file_url(paper_markdown: str) -> str | None:
    doi_match = re.search(r"\b10\.6084/m9\.(?:figshare|fgshare)\.([0-9]+)\b", paper_markdown, flags=re.IGNORECASE)
    article_id = doi_match.group(1) if doi_match else None
    file_id = _extract_first_int(r"\bfile\s*=\s*([0-9]{5,})\b", paper_markdown, flags=re.IGNORECASE)
    if article_id and file_id:
        return f"https://figshare.com/articles/dataset/{article_id}?file={file_id}"

    match = re.search(
        r"https?://(?:www\.)?(?:figshare|fgshare)\.com/[^\s)\]]+",
        paper_markdown,
        flags=re.IGNORECASE,
    )
    if not match:
        doi_match = re.search(r"\b10\.6084/m9\.(?:figshare|fgshare)\.[0-9]+\b", paper_markdown, flags=re.IGNORECASE)
        if doi_match:
            return f"https://doi.org/{_norm_url_ocr(doi_match.group(0))}"
        return None
    url = _norm_url_ocr(match.group(0).strip().rstrip(".,;"))
    if not url:
        return None
    file_id = _extract_first_int(r"\bfile\s*=\s*([0-9]{5,})\b", paper_markdown, flags=re.IGNORECASE)
    if file_id:
        base = re.sub(r"[?&]file=[0-9]+", "", url, flags=re.IGNORECASE)
        sep = "&" if "?" in base else "?"
        url = f"{base}{sep}file={file_id}"
    normed = _norm_url_ocr(url)
    has_numeric_article_id = bool(normed and re.search(r"/[0-9]{6,}(?:[/?]|$)", normed))
    if normed and (len(normed) < 35 or not has_numeric_article_id or normed.lower().endswith("/_b_")):
        doi_match = re.search(r"\b10\.6084/m9\.(?:figshare|fgshare)\.[0-9]+\b", paper_markdown, flags=re.IGNORECASE)
        if doi_match:
            return f"https://doi.org/{_norm_url_ocr(doi_match.group(0))}"
    return normed


def _strict_file_list_enabled() -> bool:
    raw = (os.getenv("P2D_STRICT_FILE_LIST") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_file_entries(values: list[str]) -> list[str]:
    allowed_ext = ("xlsx", "csv", "tsv", "ipynb", "py", "md", "txt", "json", "zip", "pdf")
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = html.unescape(str(raw or "")).strip()
        if not token:
            continue
        token = re.sub(r"\s+", " ", token)
        file_match = re.search(r"\b([A-Za-z0-9_.-]+\.(?:xlsx|csv|tsv|ipynb|py|md|txt|json|zip|pdf))\b", token, flags=re.IGNORECASE)
        normalized: str | None = None
        if file_match:
            normalized = file_match.group(1)
        elif token.lower().startswith("readme"):
            normalized = "README.md"
        elif _strict_file_list_enabled():
            normalized = None
        else:
            # Keep compact basename-like tokens in non-strict mode.
            simple = re.sub(r"[^A-Za-z0-9_.-]", "", token)
            if simple and "." in simple:
                ext = simple.rsplit(".", 1)[-1].lower()
                if ext in allowed_ext:
                    normalized = simple
        if not normalized:
            continue
        if normalized.lower() == "readme":
            normalized = "README.md"
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _extract_figshare_ids(accession: DataAccession) -> tuple[str | None, str | None]:
    article_id: str | None = None
    file_id: str | None = None
    combined = " ".join(
        [
            accession.accession_id or "",
            accession.url or "",
            accession.download_probe_url or "",
            " ".join(accession.files_listed or []),
        ]
    )
    m_article = re.search(r"\b10\.6084/m9\.figshare\.([0-9]+)\b", combined, flags=re.IGNORECASE)
    if m_article:
        article_id = m_article.group(1)
    if not article_id:
        m_article = re.search(r"/articles?/[^/]+/([0-9]+)", combined, flags=re.IGNORECASE)
        if m_article:
            article_id = m_article.group(1)
    m_file = re.search(r"[?&]file=([0-9]+)", combined, flags=re.IGNORECASE)
    if m_file:
        file_id = m_file.group(1)
    if not file_id:
        m_file = re.search(r"/files/([0-9]+)", combined, flags=re.IGNORECASE)
        if m_file:
            file_id = m_file.group(1)
    return article_id, file_id


def _extract_table_blocks(text: str) -> list[ExtractedTable]:
    blocks = re.findall(r"<table>(.*?)</table>", text, flags=re.IGNORECASE | re.DOTALL)
    extracted: list[ExtractedTable] = []
    for i, block in enumerate(blocks, start=1):
        rows = re.findall(r"<tr>(.*?)</tr>", block, flags=re.IGNORECASE | re.DOTALL)
        parsed_rows: list[list[str]] = []
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
            cleaned = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip()
                for cell in cells
            ]
            cleaned = [c for c in cleaned if c]
            if cleaned:
                parsed_rows.append(cleaned)
        if not parsed_rows:
            continue
        columns = parsed_rows[0][:20]
        row_dicts: list[dict[str, str]] = []
        for row in parsed_rows[1:]:
            normalized_row = row + [""] * max(0, len(columns) - len(row))
            row_obj: dict[str, str] = {}
            for idx, col in enumerate(columns):
                key = re.sub(r"\s+", " ", col).strip() or f"col_{idx+1}"
                row_obj[key] = normalized_row[idx].strip() if idx < len(normalized_row) else ""
            if any(v for v in row_obj.values()):
                row_dicts.append(row_obj)
        key_content: list[str] = []
        seen: set[str] = set()
        for row in parsed_rows[1:]:
            for cell in row:
                parts = re.split(r"[•▪◦●;]\s*|\s+-\s+", cell.strip())
                if len(parts) == 1:
                    parts = [cell.strip()]
                for part in parts:
                    token = re.sub(r"\s+", " ", part).strip(" -\t\r\n")
                    token = token.replace(".The ", ". The ")
                    token = token.replace(".There ", ". There ")
                    sentence_parts = [s.strip() for s in re.split(r"\.\s+", token) if s.strip()]
                    if len(sentence_parts) > 1:
                        for sent in sentence_parts:
                            if len(sent) < 2:
                                continue
                            low_sent = sent.lower()
                            if low_sent in seen:
                                continue
                            seen.add(low_sent)
                            key_content.append(sent)
                            if len(key_content) >= 40:
                                break
                        if len(key_content) >= 40:
                            break
                        continue
                    if len(token) < 2:
                        continue
                    if "inclusion criteria" in token.lower() and "exclusion criteria" in token.lower():
                        subparts = re.split(r"(?i)\b(inclusion criteria|exclusion criteria)\b", token)
                        for sub in subparts:
                            sub_clean = re.sub(r"\s+", " ", sub).strip(" -:\t\r\n")
                            if len(sub_clean) < 3:
                                continue
                            low_sub = sub_clean.lower()
                            if low_sub in seen:
                                continue
                            seen.add(low_sub)
                            key_content.append(sub_clean)
                            if len(key_content) >= 40:
                                break
                        if len(key_content) >= 40:
                            break
                        continue
                    low = token.lower()
                    if low in seen:
                        continue
                    seen.add(low)
                    key_content.append(token)
                    if len(key_content) >= 40:
                        break
            if len(key_content) >= 40:
                break
        table_prov = _locate_provenance(text, f"table {i}", source_table=f"Table {i}") or _locate_provenance(
            text, "<table>", source_table=f"Table {i}"
        )
        extracted.append(
            ExtractedTable(
                table_id=f"Table {i}",
                title=f"Extracted table {i}",
                columns=columns,
                data=row_dicts[:400],
                summary=f"Parsed {len(parsed_rows)} rows from table {i}.",
                key_content=key_content,
                provenance=table_prov,
            )
        )
    return extracted


def _extract_dataset_profile_from_text(
    paper_markdown: str,
    *,
    existing: DatasetProfile | None,
    metadata_license: str | None,
    anatomy_prisma_flow: dict[str, int],
) -> DatasetProfile | None:
    profile = existing.model_copy(deep=True) if existing else DatasetProfile()
    text = paper_markdown
    lowered = text.lower()

    if not profile.name:
        candidates = re.findall(r"#\s+([^\n]+)", paper_markdown)
        if candidates:
            profile.name = candidates[0].strip()

    if profile.record_count is None:
        profile.record_count = _extract_first_int(
            r"\b([0-9][0-9,]*)\s+(?:measures|records|rows|entries|associations)\b",
            lowered,
        )
    measures_count = _extract_first_int(r"\b([0-9][0-9,]*)\s+measures\b", lowered)
    if measures_count is not None and (profile.record_count is None or profile.record_count < measures_count):
        profile.record_count = measures_count
    dims_measures = profile.dimensions.get("measures") if isinstance(profile.dimensions, dict) else None
    if isinstance(dims_measures, int) and (profile.record_count is None or profile.record_count < dims_measures):
        profile.record_count = dims_measures
    if profile.data_point_count is None:
        profile.data_point_count = _extract_first_int(
            r"\b([0-9][0-9,]*)\s+data\s+points?\b",
            lowered,
        )
    if profile.columns is None:
        profile.columns = _extract_first_int(r"\b([0-9][0-9,]*)\s+columns?\b", lowered)

    if not profile.temporal_coverage:
        range_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\s*(?:-|–|to)\s*(19[0-9]{2}|20[0-9]{2})\b", text)
        if range_match:
            profile.temporal_coverage = f"{range_match.group(1)}-{range_match.group(2)}"

    if profile.source_corpus_size is None:
        profile.source_corpus_size = _extract_first_int(
            r"\b([0-9][0-9,]*)\s+(?:papers?|articles?)\s+(?:in(?:\s+the)?\s+)?(?:final\s+corpus|included)\b",
            lowered,
        )
    if profile.source_corpus_size and profile.record_count and profile.record_count == profile.source_corpus_size and measures_count:
        profile.record_count = measures_count

    if not profile.version:
        version_match = re.search(r"\bversion\s+([0-9]+(?:\.[0-9]+)?)\b", lowered)
        if version_match:
            profile.version = version_match.group(1)

    formats = set(profile.format or [])
    ext_hits = {
        "xlsx": r"\.xlsx\b|\bxlsx\b",
        "csv": r"\.csv\b|\bcsv\b",
        "tsv": r"\.tsv\b|\btsv\b|\btab[- ]delimited\b",
        "json": r"\.json\b|\bjson\b",
        "rdf": r"\brdf\b|\btrig\b",
    }
    for fmt, pattern in ext_hits.items():
        if re.search(pattern, lowered):
            formats.add(fmt)
    profile.format = sorted(formats)

    if not profile.license:
        profile.license = metadata_license
    if not profile.license:
        lic_match = re.search(r"\b(cc\s*by(?:[- ]?\d\.\d)?|cc0|creative commons[^,\n.]*)\b", text, flags=re.IGNORECASE)
        if lic_match:
            profile.license = re.sub(r"\s+", " ", lic_match.group(1)).strip()

    contents = list(profile.repository_contents or [])
    content_patterns = [
        r"\bREADME(?:\.md)?\b",
        r"\b[^ \n\t]+\.xlsx\b",
        r"\b[^ \n\t]+\.csv\b",
        r"\b[^ \n\t]+\.tsv\b",
        r"\b[^ \n\t]+\.ipynb\b",
        r"\b[^ \n\t]+\.py\b",
        r"\b[^ \n\t]+\.md\b",
    ]
    for pattern in content_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            token = re.sub(r"[),.;]+$", "", match.strip())
            if token and token not in contents:
                contents.append(token)
    if any(x.lower() == "sunburst_visualization_link.md" for x in contents):
        contents = [x for x in contents if x.lower() != "link.md"]
    contents = [x for x in contents if x.lower() != "link.md"]
    profile.repository_contents = contents[:80]

    existing_flow = profile.prisma_flow.as_compact_dict() if isinstance(profile.prisma_flow, PrismaFlow) else (profile.prisma_flow or {})
    prisma_flow_raw = _extract_prisma_flow(text, existing=existing_flow or anatomy_prisma_flow)
    profile.prisma_flow = _prisma_model(prisma_flow_raw)
    compact_flow = profile.prisma_flow.as_compact_dict()
    included_from_prisma = compact_flow.get("included")
    if isinstance(included_from_prisma, int) and included_from_prisma > 0:
        if profile.source_corpus_size is None or profile.source_corpus_size > included_from_prisma:
            profile.source_corpus_size = included_from_prisma

    if not isinstance(profile.dimensions, dict):
        profile.dimensions = {}
    assessment_terms = {
        "statistical": "statistical",
        "mathematical": "mathematical",
        "behavioural": "behavioural",
        "behavioral": "behavioural",
        "self-reported": "self-reported",
        "qualitative": "qualitative_or_other",
        "unspecified": "qualitative_or_other",
        "other": "qualitative_or_other",
    }
    found_assessments: set[str] = set()
    for token, label in assessment_terms.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            found_assessments.add(label)
    if found_assessments:
        profile.dimensions["assessments"] = max(int(profile.dimensions.get("assessments", 0) or 0), len(found_assessments))

    physical_keys = {"rows", "columns", "files", "bytes", "record_count", "data_point_count"}
    physical = dict(profile.physical_dimensions or {})
    conceptual = dict(profile.conceptual_dimensions or {})
    for key, value in (profile.dimensions or {}).items():
        if key in physical_keys:
            physical[key] = value
        else:
            conceptual[key] = value
    if profile.record_count is not None:
        physical["record_count"] = profile.record_count
    if profile.data_point_count is not None:
        physical["data_point_count"] = profile.data_point_count
    if profile.columns is not None:
        physical["columns"] = profile.columns
    profile.physical_dimensions = physical
    profile.conceptual_dimensions = conceptual

    if not profile.processing_pipeline_summary:
        summary_match = re.search(
            r"(workflow\..{0,500})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            profile.processing_pipeline_summary = re.sub(r"\s+", " ", summary_match.group(1)).strip()[:600]

    known_columns = [
        ("Measure", "Dataset details", "Name/identifier of a measure."),
        ("Measurement Process", "Dataset details", "How the measure was operationalized."),
        ("Principle", "Ethical principle", "Responsible AI principle targeted by the measure."),
        ("Part of the ML System", "System component", "ML lifecycle/system component assessed."),
        ("Primary Harm", "Harm taxonomy", "Primary harm category addressed."),
        ("Secondary Harm", "Harm taxonomy", "Secondary harm category addressed."),
        ("Criterion Name", "Assessment details", "Criterion title for evaluation."),
        ("Criterion Description", "Assessment details", "Description of the evaluation criterion."),
        ("Type of Assessment", "Assessment details", "Assessment modality/type."),
        ("Application Area", "Paper metadata", "Application domain of the ML system."),
        ("Purpose of ML System", "Paper metadata", "Purpose/task of the ML system."),
        ("Type of Data", "Paper metadata", "Data modality used by the ML system."),
        ("Algorithm Type", "Paper metadata", "Model/algorithm family."),
        ("Title", "Paper metadata", "Source paper title."),
        ("Publication Year", "Paper metadata", "Year of source paper publication."),
        ("DOI Link", "Paper metadata", "DOI URL/identifier for source paper."),
    ]
    col_schema = list(profile.column_schema or [])
    existing_names = {c.name.lower() for c in col_schema}
    for name, category, description in known_columns:
        if name.lower() in lowered and name.lower() not in existing_names:
            col_schema.append(DatasetColumn(name=name, category=category, description=description))
    profile.column_schema = col_schema
    if profile.columns is None and profile.column_schema:
        profile.columns = len(profile.column_schema)

    has_signal = any(
        [
            profile.record_count is not None,
            profile.data_point_count is not None,
            profile.columns is not None,
            bool(profile.temporal_coverage),
            bool(profile.source_corpus_size),
            bool(profile.repository_contents),
            bool(profile.column_schema),
        ]
    )
    return profile if has_signal else existing


def _backfill_dataset_results(
    *,
    results,
    metadata_license: str | None,
    paper_markdown: str,
    anatomy,
) -> None:
    if (results.paper_type or "").strip().lower() != "dataset_descriptor":
        return
    profile = _extract_dataset_profile_from_text(
        paper_markdown,
        existing=results.dataset_profile,
        metadata_license=metadata_license,
        anatomy_prisma_flow=anatomy.prisma_flow,
    )
    if profile:
        cleaned_schema: list[DatasetColumn] = []
        seen_schema: set[str] = set()
        for col in profile.column_schema or []:
            name = re.sub(r"\s+", " ", (col.name or "").strip())
            if not name:
                continue
            key = name.lower()
            if key in seen_schema:
                continue
            if (col.description is None or not str(col.description).strip()) and (
                col.category is None or not str(col.category).strip()
            ):
                continue
            seen_schema.add(key)
            cleaned_schema.append(
                DatasetColumn(
                    name=name,
                    category=(col.category or None),
                    description=(col.description or None),
                )
            )
        if cleaned_schema:
            profile.column_schema = cleaned_schema
        results.dataset_profile = profile
    if not results.dataset_properties and results.dataset_profile:
        p = results.dataset_profile
        props: list[DescriptiveStat] = []
        if p.record_count is not None:
            props.append(
                DescriptiveStat(
                    property="record_count",
                    variable="record_count",
                    value=str(p.record_count),
                    context="Dataset size",
                    confidence=0.9,
                    provenance=_locate_provenance(paper_markdown, "measures"),
                )
            )
        if p.data_point_count is not None:
            props.append(
                DescriptiveStat(
                    property="data_point_count",
                    variable="data_point_count",
                    value=str(p.data_point_count),
                    context="Total extracted datapoints",
                    confidence=0.9,
                    provenance=_locate_provenance(paper_markdown, "data points"),
                )
            )
        if p.columns is not None:
            props.append(
                DescriptiveStat(
                    property="columns",
                    variable="columns",
                    value=str(p.columns),
                    context="Dataset schema width",
                    confidence=0.88,
                    provenance=_locate_provenance(paper_markdown, "columns"),
                )
            )
        if p.temporal_coverage:
            props.append(
                DescriptiveStat(
                    property="temporal_coverage",
                    variable="temporal_coverage",
                    value=p.temporal_coverage,
                    context="Time span covered",
                    confidence=0.85,
                    provenance=_locate_provenance(paper_markdown, p.temporal_coverage),
                )
            )
        if p.source_corpus_size is not None:
            props.append(
                DescriptiveStat(
                    property="source_corpus_size",
                    variable="source_corpus_size",
                    value=str(p.source_corpus_size),
                    context="Source papers included",
                    confidence=0.88,
                    provenance=_locate_provenance(paper_markdown, "included"),
                )
            )
        p_flow = p.prisma_flow.as_compact_dict() if isinstance(p.prisma_flow, PrismaFlow) else (p.prisma_flow or {})
        if p_flow:
            props.append(
                DescriptiveStat(
                    property="prisma_flow",
                    variable="prisma_flow",
                    value=", ".join([f"{k}:{v}" for k, v in sorted(p_flow.items())]),
                    context="Study selection flow counts",
                    confidence=0.86,
                    provenance=_locate_provenance(paper_markdown, "PRISMA"),
                )
            )
        results.dataset_properties = props

    parsed_tables = _extract_table_blocks(paper_markdown)
    if parsed_tables:
        if not results.tables_extracted:
            results.tables_extracted = parsed_tables
        else:
            by_id = {t.table_id.lower(): t for t in results.tables_extracted}
            for parsed in parsed_tables:
                existing = by_id.get(parsed.table_id.lower())
                if not existing:
                    results.tables_extracted.append(parsed)
                    continue
                if not existing.key_content and parsed.key_content:
                    existing.key_content = parsed.key_content
                if not existing.columns and parsed.columns:
                    existing.columns = parsed.columns
                if not existing.data and parsed.data:
                    existing.data = parsed.data


def _derive_code_repositories(
    paper_markdown: str,
    data_accessions: list[DataAccession],
    related_resources: list[RelatedResource],
    dataset_profile: DatasetProfile | None,
) -> list[str]:
    candidates: list[str] = []
    for match in re.findall(r"https?://[^\s)\]]+", paper_markdown, flags=re.IGNORECASE):
        url = _norm_url_ocr(match.strip().rstrip(".,;"))
        if not url:
            continue
        low = url.lower()
        if any(token in low for token in ("github.com", "gitlab.com", "bitbucket.org")):
            candidates.append(url)
    for accession in data_accessions:
        if accession.url and (accession.data_format in {"py", "ipynb"} or "figshare" in accession.url.lower()):
            candidates.append(accession.url)
    for resource in related_resources:
        if resource.url and resource.type in {"tool", "visualization"}:
            low = resource.url.lower()
            if any(token in low for token in ("github.com", "gitlab.com", "bitbucket.org", "figshare.com")):
                candidates.append(resource.url)
    for name in (dataset_profile.repository_contents if dataset_profile else []):
        low = name.lower()
        if low.endswith(".py") or low.endswith(".ipynb"):
            candidates.append(f"bundled_file:{name}")
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _split_code_repositories(code_repositories: list[str]) -> tuple[list[str], list[str]]:
    vcs: list[str] = []
    archival: list[str] = []
    for item in code_repositories:
        low = item.lower()
        if any(token in low for token in ("github.com", "gitlab.com", "bitbucket.org", "svn")):
            vcs.append(item)
        elif any(token in low for token in ("figshare", "zenodo", "dataverse", "dryad", "osf", "bundled_file:")):
            archival.append(item)
        else:
            archival.append(item)
    return vcs, archival


def _clean_placeholder_list(values: list[str], *, drop_journal: str | None = None) -> list[str]:
    blocked = {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "unspecified",
        "multi_species(total=0)",
    }
    if drop_journal:
        blocked.add(drop_journal.strip().lower())
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        token = re.sub(r"\s+", " ", str(v or "").strip())
        if not token:
            continue
        low = token.lower()
        if low in blocked:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(token)
    return out


def _normalize_methods_for_domain(methods, domain_profile: str) -> None:
    if domain_profile != "bio":
        methods.organisms = []
        methods.cell_types = []
    if domain_profile == "meta_research":
        cleaned: list[str] = []
        for assay in methods.assay_types:
            low = assay.lower()
            if any(k in low for k in ("scoping review", "systematic review", "meta-analysis", "statistical")):
                cleaned.append(assay)
        methods.assay_types = cleaned or methods.assay_types
    mapped: list[AssayTypeMapping] = []
    seen: set[str] = set()
    for assay in methods.assay_types:
        m = _assay_mapping_for(assay)
        key = f"{m.raw.lower()}::{m.mapped_term.lower()}"
        if key in seen:
            continue
        seen.add(key)
        mapped.append(m)
    methods.assay_type_mappings = mapped


def _fallback_ontology_map(raw: str) -> tuple[str, str | None, str | None]:
    low = raw.lower()
    rules = [
        ("manual literature", "literature search", "OBI:0000275", "OBI"),
        ("literature search", "literature search", "OBI:0000275", "OBI"),
        ("curation", "data curation", "NCIT:C153191", "NCIT"),
        ("rdf", "resource description framework", "NCIT:C85873", "NCIT"),
        ("statistical", "statistical analysis", "NCIT:C25656", "NCIT"),
        ("scoping review", "scoping review", "NCIT:C17649", "NCIT"),
        ("scoping_review", "scoping review", "NCIT:C17649", "NCIT"),
        ("systematic review", "systematic review", "NCIT:C40514", "NCIT"),
        ("systematic_review", "systematic review", "NCIT:C40514", "NCIT"),
    ]
    for needle, mapped, oid, vocab in rules:
        if needle in low:
            return mapped, oid, vocab
    mapped = re.sub(r"\s+", " ", raw).strip().lower()[:80] or "unspecified assay"
    return mapped, None, None


def _resolve_assay_ontology_mappings(methods) -> None:
    resolved: list[AssayTypeMapping] = []
    seen: set[str] = set()
    for m in methods.assay_type_mappings:
        raw = (m.raw or "").strip() or (m.mapped_term or "").strip()
        mapped_term = (m.mapped_term or "").strip()
        ontology_id = (m.ontology_id or "").strip() or None
        vocabulary = (m.vocabulary or "").strip() or None
        if not ontology_id:
            mapped_term, ontology_id, vocabulary = _fallback_ontology_map(raw)
        key = f"{raw.lower()}::{(ontology_id or '').lower()}::{mapped_term.lower()}"
        if key in seen:
            continue
        seen.add(key)
        resolved.append(
            AssayTypeMapping(
                raw=raw,
                mapped_term=mapped_term,
                ontology_id=ontology_id,
                vocabulary=vocabulary,
            )
        )
    methods.assay_type_mappings = resolved


def _partition_results_findings_vs_properties(results) -> None:
    props = list(results.dataset_properties or [])
    finds = list(results.experimental_findings or [])
    if not props and not finds:
        return
    prop_signatures = {
        f"{(p.property or '').strip().lower()}::{(p.value or '').strip().lower()}" for p in props
    }
    filtered_findings: list[ExperimentalFinding] = []
    for f in finds:
        metric = (f.metric or "").strip().lower()
        value = (f.value or "").strip().lower()
        sig = f"{metric}::{value}"
        unit = (f.unit or "").strip().lower()
        # Static artifact-size facts should not stay in findings.
        is_static = any(
            k in metric for k in ("row", "record", "column", "file size", "bytes", "dataset size", "source corpus")
        ) or unit in {"rows", "bytes", "columns"}
        if sig in prop_signatures or is_static:
            continue
        filtered_findings.append(f)
    # If no scientific findings remain, keep percent/p-value style items from props.
    if not filtered_findings:
        for p in props:
            unit = (p.unit or "").strip().lower()
            value = (p.value or "").strip().lower()
            if "%" in value or "p=" in value or unit in {"percent", "p-value"}:
                filtered_findings.append(
                    ExperimentalFinding(
                        claim=f"{p.property} = {p.value}",
                        metric=p.variable or p.property,
                        value=p.value,
                        unit=p.unit,
                        statistical_significance=p.statistical_significance,
                        effect_size=None,
                        confidence_interval=None,
                        comparison=None,
                        context=p.context,
                        confidence=p.confidence,
                        provenance=p.provenance,
                    )
                )
    results.experimental_findings = filtered_findings


def _derive_keywords_from_text(paper_markdown: str, metadata_title: str) -> list[str]:
    text = f"{metadata_title}\n{paper_markdown}".lower()
    vocab = [
        "responsible ai",
        "ethical principles",
        "scoping review",
        "measurement",
        "fairness",
        "transparency",
        "privacy",
        "trust",
        "ai governance",
        "sociotechnical harms",
        "dataset",
    ]
    out: list[str] = []
    for term in vocab:
        if term in text:
            out.append(term)
    return out[:10]


def _attach_provenance_defaults(paper_markdown: str, methods, results) -> None:
    def _normalize_prov(prov: Provenance | None, seed: str, source_table: str | None = None) -> Provenance | None:
        if prov is not None and (prov.text_segment or "").strip():
            seg = str(prov.text_segment).strip()
            if seg in paper_markdown:
                return prov
        return _locate_provenance(paper_markdown, seed, source_table=source_table)

    for idx, step in enumerate(methods.experimental_design_steps, start=1):
        seed = step.action or step.context or f"step {idx}"
        step.provenance = _normalize_prov(step.provenance, seed)
    for finding in results.experimental_findings:
        seed = finding.value or finding.claim
        finding.provenance = _normalize_prov(finding.provenance, seed)
    for prop in results.dataset_properties:
        seed = prop.value or prop.property
        prop.provenance = _normalize_prov(prop.provenance, seed)
    for table in results.tables_extracted:
        table.provenance = _normalize_prov(table.provenance, table.table_id, source_table=table.table_id)


def _keywords_need_repair(values: list[str]) -> bool:
    generic = {"interdisciplinary", "data_resource", "biology", "general_science", "dataset"}
    if len(values) < 3:
        return True
    low = {v.strip().lower() for v in values if v.strip()}
    return low.issubset(generic)


def _prune_related_resources(
    resources: list[RelatedResource],
    *,
    code_repositories: list[str],
    data_accession_urls: list[str],
    paper_doi: str | None,
) -> list[RelatedResource]:
    code_keys = {c.strip().lower() for c in code_repositories}
    data_keys = {u.strip().lower() for u in data_accession_urls if u and u.strip()}
    paper_doi_key = (paper_doi or "").strip().lower()
    out: list[RelatedResource] = []
    for r in resources:
        u = (r.url or "").strip().lower()
        n = (r.name or "").strip().lower()
        is_code_like = r.type == "tool" and (".py" in u or ".ipynb" in u or "github" in u or "gitlab" in u or "figshare" in u)
        if u and u in code_keys:
            continue
        if u and u in data_keys:
            continue
        if paper_doi_key and (paper_doi_key in u):
            continue
        if n.startswith("bundled_file:") and any(n == ck for ck in code_keys):
            continue
        if is_code_like:
            continue
        out.append(r)
    return out


@dataclass
class PipelineArtifacts:
    record: PaperRecord
    retrieval_report_markdown: str
    retrieval_log_markdown: str
    step_timings_seconds: dict[str, float]
    pipeline_duration_seconds: float
    quality_notes: str | None = None


async def run_pipeline(paper_markdown: str, fast_mode: bool = False) -> PipelineArtifacts:
    paper_markdown = _clean_input_text(paper_markdown)
    pipeline_start = perf_counter()
    step_timings_seconds: dict[str, float] = {}

    reset_events()
    log_event("pipeline.start", {"chars": len(paper_markdown)})

    step_start = perf_counter()
    anatomy = await run_anatomy_agent(paper_markdown)
    step_timings_seconds["anatomy"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "anatomy", "seconds": step_timings_seconds["anatomy"]})
    _print_step_timing("anatomy", step_timings_seconds["anatomy"])

    step_start = perf_counter()
    try:
        archetype = await run_archetype_agent(paper_markdown)
    except Exception:  # noqa: BLE001
        inferred = _infer_paper_type("", paper_markdown)
        archetype_map = {
            "dataset_descriptor": "dataset",
            "methods": "methodology",
            "review": "review",
            "meta_analysis": "review",
            "experimental": "experimental_study",
            "commentary": "commentary",
        }
        archetype = ArchetypeOutput(
            paper_archetype=archetype_map.get(inferred, "experimental_study"),
            confidence=0.5,
            rationale="fallback heuristic",
        )
    step_timings_seconds["archetype"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "archetype", "seconds": step_timings_seconds["archetype"]})
    _print_step_timing("archetype", step_timings_seconds["archetype"])

    anatomy_guidance = _anatomy_guidance_blob(
        anatomy.sections,
        anatomy.tables,
        anatomy.figures,
        anatomy.urls,
    )

    step_start = perf_counter()
    metadata = await run_metadata_agent(paper_markdown, guidance=anatomy_guidance)
    if not is_valid_category_subcategory(metadata.category, metadata.subcategory):
        log_event(
            "pipeline.taxonomy.flagged_invalid",
            {
                "category": metadata.category,
                "subcategory": metadata.subcategory,
            },
        )
        if not fast_mode:
            taxonomy_guidance = (
                "Field/subfield invalid. Select one exact pair from allowed taxonomy only."
            )
            repaired = await run_metadata_agent(paper_markdown, guidance=taxonomy_guidance)
            metadata.category = repaired.category
            metadata.subcategory = repaired.subcategory
    metadata.category, metadata.subcategory = normalize_category_subcategory(
        metadata.category, metadata.subcategory
    )
    if _is_blank(metadata.paper_type):
        metadata.paper_type = _infer_paper_type(metadata.title, paper_markdown)
    if _is_blank(metadata.paper_archetype):
        metadata.paper_archetype = archetype.paper_archetype
    domain_profile = _infer_domain_profile(metadata.category, metadata.paper_type, paper_markdown)
    metadata.journal = _normalize_venue_name(metadata.journal)
    metadata.license = _normalize_license_to_spdx(metadata.license)
    step_timings_seconds["metadata"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "metadata", "seconds": step_timings_seconds["metadata"]})
    _print_step_timing("metadata", step_timings_seconds["metadata"])

    step_start = perf_counter()
    methods_guidance = (
        f"{anatomy_guidance}\n"
        f"Domain profile: {domain_profile}\n"
        "Return atomic experimental_design_steps; avoid domain-mismatched placeholders."
    )
    methods = await run_methods_agent(paper_markdown, guidance=methods_guidance)
    step_timings_seconds["methods"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "methods", "seconds": step_timings_seconds["methods"]})
    _print_step_timing("methods", step_timings_seconds["methods"])

    step_start = perf_counter()
    results_guidance = (
        f"{anatomy_guidance}\n"
        f"PRISMA candidates: {anatomy.prisma_flow}\n"
        "Sub-extraction mode:\n"
        "- TableExtractor: return tables_extracted with row-level data\n"
        "- TextClaimExtractor: capture numeric claims with values and units\n"
        "- Reconciliation: prefer values corroborated by table/text overlap\n"
    )
    results = await run_results_agent(
        paper_markdown,
        paper_type=metadata.paper_type,
        guidance=results_guidance,
    )
    if _is_blank(results.paper_type):
        results.paper_type = metadata.paper_type
    _backfill_dataset_results(
        results=results,
        metadata_license=metadata.license,
        paper_markdown=paper_markdown,
        anatomy=anatomy,
    )
    results.tables_extracted = _normalize_table_topology(results.tables_extracted)
    if not results.experimental_findings:
        results.experimental_findings = _extract_quantitative_findings_from_text(
            paper_markdown,
            existing=results.experimental_findings,
        )
    if not results.experimental_findings and results.dataset_properties:
        results.experimental_findings = _derive_findings_from_dataset_properties(
            results.dataset_properties
        )
    prisma_for_steps = results.dataset_profile.prisma_flow if results.dataset_profile else _prisma_model(anatomy.prisma_flow)
    if not methods.experimental_design_steps:
        methods.experimental_design_steps = _extract_experimental_design_steps(
            paper_markdown,
            methods.experimental_design,
            prisma_for_steps,
        )
    _normalize_methods_for_domain(methods, domain_profile)
    _resolve_assay_ontology_mappings(methods)
    _reclassify_dataset_counts_to_sample_sizes(methods, results)
    _partition_results_findings_vs_properties(results)
    _derive_sample_size_records(methods, paper_markdown)
    _set_missingness(methods, metadata.paper_archetype or metadata.paper_type)
    _link_tool_entities(methods)
    _build_findings_block(results)
    _export_large_tables(results.tables_extracted)
    step_timings_seconds["results"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "results", "seconds": step_timings_seconds["results"]})
    _print_step_timing("results", step_timings_seconds["results"])

    step_start = perf_counter()
    if fast_mode:
        data_availability = DataAvailabilityOutput(
            data_accessions=[],
            related_resources=[],
            data_availability=DataAvailabilityReport(
                overall_status="not_checked",
                claimed_repositories=[],
                verified_repositories=[],
                discrepancies=["fast_mode: data availability deep checks skipped"],
                notes="Fast mode enabled: data availability checks skipped.",
                check_status="not_checked",
            ),
        )
    else:
        try:
            data_availability = await run_data_availability_agent(
                paper_markdown,
                paper_type=metadata.paper_type,
                guidance=anatomy_guidance,
            )
        except Exception as exc:  # noqa: BLE001
            log_event("agent.data_availability.error", {"error": str(exc)})
            data_availability = await fallback_data_availability_from_text(
                paper_markdown=paper_markdown,
                paper_type=metadata.paper_type,
                reason=str(exc),
            )
    step_timings_seconds["data_availability"] = perf_counter() - step_start
    log_event(
        "pipeline.step_timing",
        {"step": "data_availability", "seconds": step_timings_seconds["data_availability"]},
    )
    _print_step_timing("data_availability", step_timings_seconds["data_availability"])
    _apply_url_canonicalization(
        data_accessions=data_availability.data_accessions,
        related_resources=getattr(data_availability, "related_resources", []),
    )
    _normalize_accession_identifiers(data_availability.data_accessions)
    if results.dataset_profile and results.dataset_profile.repository_contents:
        profile_contents = _normalize_file_entries(results.dataset_profile.repository_contents)
        results.dataset_profile.repository_contents = profile_contents
        figshare_file_url = _extract_figshare_file_url(paper_markdown)
        for accession in data_availability.data_accessions:
            repo = (accession.repository or "").lower()
            if "figshare" in repo:
                article_id, file_id = _extract_figshare_ids(accession)
                if figshare_file_url:
                    accession.url = figshare_file_url
                elif article_id and file_id:
                    accession.url = f"https://figshare.com/articles/dataset/{article_id}?file={file_id}"
                elif article_id:
                    accession.url = f"https://doi.org/10.6084/m9.figshare.{article_id}"
                files = list(accession.files_listed or [])
                seen = {f.lower() for f in files}
                for item in profile_contents:
                    key = item.strip().lower()
                    if not key or key in seen:
                        continue
                    files.append(item.strip())
                    seen.add(key)
                if any(x.lower() == "sunburst_visualization_link.md" for x in files):
                    files = [x for x in files if x.lower() != "link.md"]
                files = [x for x in files if x.lower() != "link.md"]
                accession.files_listed = _normalize_file_entries(files)[:120]
                accession.file_count = max(accession.file_count or 0, len(accession.files_listed))

    if fast_mode:
        # Lightweight QC placeholder in fast mode.
        from src.schemas.models import QualityCheckOutput

        quality_check = QualityCheckOutput(
            missing_fields=[],
            suspicious_empty_fields=[],
            should_retry=False,
            retry_instructions=[],
            notes="Fast mode: quality-control agent skipped.",
        )
    else:
        step_start = perf_counter()
        quality_check = await run_quality_control_agent(
            paper_markdown=paper_markdown,
            metadata=metadata,
            methods=methods,
            results=results,
            data_availability=data_availability.data_availability,
        )
        step_timings_seconds["quality_control"] = perf_counter() - step_start
        log_event(
            "pipeline.step_timing",
            {"step": "quality_control", "seconds": step_timings_seconds["quality_control"]},
        )
        _print_step_timing("quality_control", step_timings_seconds["quality_control"])

    if not fast_mode and quality_check.should_retry and quality_check.retry_instructions:
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
                methods = await run_methods_agent(paper_markdown, guidance=f"{reason}\n{anatomy_guidance}")
            elif agent_name == "results":
                results = await run_results_agent(
                    paper_markdown,
                    paper_type=metadata.paper_type,
                    guidance=f"{reason}\n{results_guidance}",
                )
            elif agent_name in {"data_availability", "data-availability"}:
                data_availability = await run_data_availability_agent(
                    paper_markdown,
                    paper_type=metadata.paper_type,
                    guidance=f"{reason}\n{anatomy_guidance}",
                )
            log_event("pipeline.quality_retry.step", {"agent": agent_name, "reason": reason})
        log_event("pipeline.quality_retry.end", {"notes": quality_check.notes})

    needs_enrichment = (
        (not metadata.doi)
        or (not metadata.pmid)
        or (not metadata.journal)
        or (metadata.journal.strip().lower() == "scientific data")
    )
    if not fast_mode and needs_enrichment:
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
        metadata.journal = _normalize_venue_name(metadata.journal)
        metadata.license = _normalize_license_to_spdx(metadata.license)
        log_event("pipeline.metadata_enrichment.applied", {"notes": enrichment.notes})

    missing_before_repair = _missing_key_metadata_fields(
        title=metadata.title,
        authors=metadata.authors,
        journal=metadata.journal,
        publication_date=metadata.publication_date,
        keywords=metadata.keywords,
        paper_type=metadata.paper_type,
    )
    if not fast_mode and missing_before_repair:
        guidance = (
            "Repair metadata completeness. Fill these fields with best available evidence from paper text "
            f"and tools: {', '.join(missing_before_repair)}. "
            "Prefer precise values; do not leave these as null/empty."
        )
        repaired = await run_metadata_agent(paper_markdown, guidance=guidance)
        repaired.category, repaired.subcategory = normalize_category_subcategory(repaired.category, repaired.subcategory)
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
    metadata.journal = _normalize_venue_name(metadata.journal)
    if _is_blank(metadata.publication_date):
        metadata.publication_date = _extract_publication_date(paper_markdown) or _extract_year_hint(paper_markdown) or "Unknown"
    elif metadata.publication_date and re.fullmatch(r"\d{4}", metadata.publication_date.strip()):
        precise_date = _extract_publication_date(paper_markdown)
        if precise_date:
            metadata.publication_date = precise_date
    metadata.keywords = _clean_placeholder_list(metadata.keywords, drop_journal=metadata.journal)
    methods.organisms = _clean_placeholder_list(methods.organisms)
    methods.cell_types = _clean_placeholder_list(methods.cell_types)
    methods.assay_types = _clean_placeholder_list(methods.assay_types)
    _normalize_methods_for_domain(methods, _infer_domain_profile(metadata.category, metadata.paper_type, paper_markdown))
    _resolve_assay_ontology_mappings(methods)
    if not metadata.keywords:
        metadata.keywords = _derive_keywords(
            metadata.keywords,
            metadata.category,
            metadata.subcategory,
            methods.assay_types,
            methods.organisms,
            metadata.journal,
        )
    if _keywords_need_repair(metadata.keywords):
        metadata.keywords = _derive_keywords_from_text(paper_markdown, metadata.title)
    metadata.keywords = _clean_placeholder_list(metadata.keywords, drop_journal=metadata.journal)
    if not metadata.keywords:
        metadata.keywords = _derive_keywords_from_text(paper_markdown, metadata.title)
    metadata.keywords = _clean_placeholder_list(metadata.keywords, drop_journal=metadata.journal)
    if not metadata.keywords:
        metadata.keywords = ["dataset"]
    if _is_blank(metadata.paper_type):
        metadata.paper_type = _infer_paper_type(metadata.title, paper_markdown)
    if _is_blank(metadata.paper_archetype):
        metadata.paper_archetype = archetype.paper_archetype
    metadata.publication_status = _normalize_publication_status(metadata.publication_date) or metadata.publication_status
    if _is_blank(results.paper_type):
        results.paper_type = metadata.paper_type
    _attach_provenance_defaults(paper_markdown, methods, results)
    _partition_results_findings_vs_properties(results)
    _derive_sample_size_records(methods, paper_markdown)
    _set_missingness(methods, metadata.paper_archetype or metadata.paper_type)
    _link_tool_entities(methods)
    _build_findings_block(results)
    _export_large_tables(results.tables_extracted)

    code_repositories = _derive_code_repositories(
        paper_markdown,
        data_accessions=data_availability.data_accessions,
        related_resources=getattr(data_availability, "related_resources", []),
        dataset_profile=results.dataset_profile,
    )
    cleaned_related_resources = _prune_related_resources(
        list(getattr(data_availability, "related_resources", [])),
        code_repositories=code_repositories,
        data_accession_urls=[a.url or "" for a in data_availability.data_accessions],
        paper_doi=metadata.doi,
    )
    for resource in cleaned_related_resources:
        if (resource.url or "").strip().lower() in {"https://render.com", "https://render.com/"}:
            resource.url = None
            desc = (resource.description or "").strip()
            resource.description = (desc + " " if desc else "") + "Hosting provider identified, but deep visualization URL not resolved from source."
            data_availability.data_availability.discrepancies.append(
                "Visualization deep link unresolved: only hosting provider domain was found."
            )
            break
    vcs_repositories, archival_repositories = _split_code_repositories(code_repositories)
    data_assets = _derive_data_assets(
        data_accessions=data_availability.data_accessions,
        dataset_profile=results.dataset_profile,
        paper_markdown=paper_markdown,
    )

    synthesis_input = SynthesisInput(
        metadata=metadata,
        methods=methods,
        results=results,
        data_accessions=data_availability.data_accessions,
        data_assets=data_assets,
        data_availability=data_availability.data_availability,
        code_repositories=vcs_repositories,
        vcs_repositories=vcs_repositories,
        archival_repositories=archival_repositories,
        code_available=bool(vcs_repositories or archival_repositories),
        related_resources=cleaned_related_resources,
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

    # Keep synthesis for narrative artifacts, but enforce canonical structured fields
    # from deterministic pipeline outputs to avoid model drift in final JSON.
    synthesis.record.metadata = metadata
    synthesis.record.methods = methods
    synthesis.record.results = results
    synthesis.record.data_accessions = data_availability.data_accessions
    synthesis.record.data_assets = data_assets
    synthesis.record.data_availability = data_availability.data_availability
    synthesis.record.code_repositories = vcs_repositories
    synthesis.record.vcs_repositories = vcs_repositories
    synthesis.record.archival_repositories = archival_repositories
    synthesis.record.code_available = bool(vcs_repositories or archival_repositories)
    synthesis.record.related_resources = cleaned_related_resources

    confidence = compute_extraction_confidence(
        metadata=metadata,
        methods=methods,
        results=results,
        data_availability=data_availability.data_availability,
        quality_check=quality_check,
    )
    synthesis.record.extraction_confidence = confidence.score
    synthesis.record.extraction_confidence_breakdown = confidence.as_dict()
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
