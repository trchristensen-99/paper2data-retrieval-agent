from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from agents import Agent

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
    DataAccession,
    DataAvailabilityReport,
    DatasetColumn,
    DatasetProfile,
    DescriptiveStat,
    ExtractedTable,
    PaperRecord,
    RelatedResource,
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
    if "citation_review_records" not in flow:
        cite_alt = _extract_first_int(r"citation[^\n]{0,80}\b([0-9][0-9,]*)\b", lowered)
        if cite_alt is not None:
            flow["citation_review_records"] = cite_alt
    if "expert_records" not in flow:
        expert_alt = _extract_first_int(r"expert[^\n]{0,80}\b([0-9][0-9,]*)\b", lowered)
        if expert_alt is not None:
            flow["expert_records"] = expert_alt
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
    return flow


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
        extracted.append(
            ExtractedTable(
                table_id=f"Table {i}",
                title=f"Extracted table {i}",
                columns=columns,
                summary=f"Parsed {len(parsed_rows)} rows from table {i}.",
                key_content=key_content,
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

    prisma_flow = _extract_prisma_flow(text, existing=profile.prisma_flow or anatomy_prisma_flow)
    profile.prisma_flow = prisma_flow
    included_from_prisma = prisma_flow.get("included")
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

    if not profile.processing_pipeline_summary:
        summary_match = re.search(
            r"(workflow\..{0,500})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            profile.processing_pipeline_summary = re.sub(r"\s+", " ", summary_match.group(1)).strip()[:600]

    known_columns = [
        "Measure",
        "Measurement Process",
        "Principle",
        "Part of the ML System",
        "Primary Harm",
        "Secondary Harm",
        "Criterion Name",
        "Criterion Description",
        "Type of Assessment",
        "Application Area",
        "Purpose of ML System",
        "Type of Data",
        "Algorithm Type",
        "Title",
        "Publication Year",
        "DOI Link",
    ]
    col_schema = list(profile.column_schema or [])
    existing_names = {c.name.lower() for c in col_schema}
    for name in known_columns:
        if name.lower() in lowered and name.lower() not in existing_names:
            col_schema.append(DatasetColumn(name=name))
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
        results.dataset_profile = profile
    if not results.dataset_properties and results.dataset_profile:
        p = results.dataset_profile
        props: list[DescriptiveStat] = []
        if p.record_count is not None:
            props.append(DescriptiveStat(property="record_count", value=str(p.record_count), context="Dataset size"))
        if p.data_point_count is not None:
            props.append(DescriptiveStat(property="data_point_count", value=str(p.data_point_count), context="Total extracted datapoints"))
        if p.columns is not None:
            props.append(DescriptiveStat(property="columns", value=str(p.columns), context="Dataset schema width"))
        if p.temporal_coverage:
            props.append(DescriptiveStat(property="temporal_coverage", value=p.temporal_coverage, context="Time span covered"))
        if p.source_corpus_size is not None:
            props.append(DescriptiveStat(property="source_corpus_size", value=str(p.source_corpus_size), context="Source papers included"))
        if p.prisma_flow:
            props.append(
                DescriptiveStat(
                    property="prisma_flow",
                    value=", ".join([f"{k}:{v}" for k, v in sorted(p.prisma_flow.items())]),
                    context="Study selection flow counts",
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
    pipeline_start = perf_counter()
    step_timings_seconds: dict[str, float] = {}

    reset_events()
    log_event("pipeline.start", {"chars": len(paper_markdown)})

    step_start = perf_counter()
    anatomy = await run_anatomy_agent(paper_markdown)
    step_timings_seconds["anatomy"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "anatomy", "seconds": step_timings_seconds["anatomy"]})
    _print_step_timing("anatomy", step_timings_seconds["anatomy"])

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
    metadata.journal = _normalize_venue_name(metadata.journal)
    step_timings_seconds["metadata"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "metadata", "seconds": step_timings_seconds["metadata"]})
    _print_step_timing("metadata", step_timings_seconds["metadata"])

    step_start = perf_counter()
    methods = await run_methods_agent(paper_markdown, guidance=anatomy_guidance)
    step_timings_seconds["methods"] = perf_counter() - step_start
    log_event("pipeline.step_timing", {"step": "methods", "seconds": step_timings_seconds["methods"]})
    _print_step_timing("methods", step_timings_seconds["methods"])

    step_start = perf_counter()
    results_guidance = (
        f"{anatomy_guidance}\n"
        f"PRISMA candidates: {anatomy.prisma_flow}\n"
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
    if results.dataset_profile and results.dataset_profile.repository_contents:
        profile_contents = results.dataset_profile.repository_contents
        figshare_file_url = _extract_figshare_file_url(paper_markdown)
        for accession in data_availability.data_accessions:
            repo = (accession.repository or "").lower()
            if "figshare" in repo:
                if figshare_file_url:
                    accession.url = figshare_file_url
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
                accession.files_listed = files[:120]
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
    if _is_blank(results.paper_type):
        results.paper_type = metadata.paper_type

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

    synthesis_input = SynthesisInput(
        metadata=metadata,
        methods=methods,
        results=results,
        data_accessions=data_availability.data_accessions,
        data_availability=data_availability.data_availability,
        code_repositories=code_repositories,
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
    synthesis.record.data_availability = data_availability.data_availability
    synthesis.record.code_repositories = code_repositories
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
