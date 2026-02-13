from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field
from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import DataAccession, DataAvailabilityReport, RelatedResource
from src.tools.file_tools import list_ftp_files
from src.tools.geo_tools import (
    check_geo_accession,
    check_geo_accession_request,
    check_sra_accession,
    check_sra_accession_request,
)
from src.tools.http_tools import check_url, check_url_request
from src.tools.repository_tools import (
    check_figshare_record,
    check_figshare_record_request,
    check_zenodo_record,
    check_zenodo_record_request,
    estimate_download_time,
    estimate_download_time_request,
)
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry


class DataAvailabilityOutput(BaseModel):
    data_accessions: list[DataAccession] = Field(default_factory=list)
    related_resources: list[RelatedResource] = Field(default_factory=list)
    data_availability: DataAvailabilityReport


DATA_AVAILABILITY_PROMPT = """Identify all data accessions and repository links in the paper.
For each relevant accession/URL, use tools to verify accessibility:
- check_url for web links
- check_geo_accession for GEO IDs
- check_sra_accession for SRA IDs
- list_ftp_files for FTP resources
- check_zenodo_record/check_figshare_record for repository file manifests
- estimate_download_time for transfer speed estimation

Compare claimed availability against verified availability and record discrepancies.

Critical classification rules:
- Only include entries in `data_accessions` that represent actual downloadable/queryable data assets.
- Use `category=primary_dataset` only for data produced/curated by this paper.
- Use `category=supplementary_data` for additional downloadable artifacts tied to this paper.
- Use `category=external_reference` only for cited resources not produced for this paper.
- Do NOT treat general tools/platforms/search portals (e.g., Jupyter, Render, Covidence, Web of Science) as datasets.

Output rules:
- Populate fields with extracted data values.
- Do NOT return JSON schema definitions.
- Put non-dataset links/resources in `related_resources` with type:
  visualization | tool | standard | related_dataset.
"""


data_availability_agent = Agent(
    name="data_availability_agent",
    model=MODELS.data_availability,
    instructions=DATA_AVAILABILITY_PROMPT,
    tools=[
        check_url,
        check_geo_accession,
        check_sra_accession,
        list_ftp_files,
        check_zenodo_record,
        check_figshare_record,
        estimate_download_time,
    ],
    output_type=AgentOutputSchema(DataAvailabilityOutput, strict_json_schema=False),
)

data_availability_repair_agent = Agent(
    name="data_availability_repair_agent",
    model=MODELS.data_availability,
    instructions=(
        DATA_AVAILABILITY_PROMPT
        + "\n\nReturn only populated JSON data."
        + "\nDo NOT output schema descriptors like $defs, properties, title, type, items."
    ),
)


def _normalize_repo_name(accession: DataAccession) -> str:
    return accession.repository.strip().lower()


def _norm_url_ocr(url: str | None) -> str | None:
    if not url:
        return url
    fixed = url.strip()
    fixed = re.sub(r"https?://(?:www\.)?fgshare\.com", "https://figshare.com", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\b10\.6084/m9\.fgshare\.", "10.6084/m9.figshare.", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"([?&])fle=([0-9]+)", r"\1file=\2", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"([?&])file=([0-9]{8})[0-9]+", r"\1file=\2", fixed, flags=re.IGNORECASE)
    return fixed


def _repair_url_candidates(url: str | None) -> list[str]:
    base = (url or "").strip()
    if not base:
        return []
    candidates: list[str] = []
    variants = [
        base,
        _norm_url_ocr(base) or base,
        re.sub(r"([?&])fle=([0-9]+)", r"\1file=\2", base, flags=re.IGNORECASE),
        re.sub(r"\bfgshare\b", "figshare", base, flags=re.IGNORECASE),
    ]
    for item in variants:
        clean = _norm_url_ocr(item)
        if not clean:
            continue
        if clean not in candidates:
            candidates.append(clean)
    return candidates


def _sanitize_reference_noise(resources: list[RelatedResource], paper_markdown: str) -> list[RelatedResource]:
    out: list[RelatedResource] = []
    main_doi_match = re.search(r"\b10\.\d{4,9}/[^\s)]+", paper_markdown, flags=re.IGNORECASE)
    main_doi = main_doi_match.group(0).lower().rstrip(".,;") if main_doi_match else ""
    for r in resources:
        url = (r.url or "").strip()
        low = url.lower()
        if low and main_doi and main_doi in low:
            continue
        if "doi.org/" in low and "/s41597-" in low and "nature.com" in low:
            continue
        out.append(r)
    return out


def _infer_data_format(accession: DataAccession) -> str | None:
    candidates = []
    if accession.url:
        candidates.append(accession.url.lower())
    candidates.extend([str(x).lower() for x in (accession.files_listed or [])])
    joined = " ".join(candidates)
    ext_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".xlsx": "xlsx",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".fastq": "fastq",
        ".fastq.gz": "fastq",
        ".bam": "bam",
        ".vcf": "vcf",
        ".txt": "txt",
        ".parquet": "parquet",
    }
    for ext, fmt in ext_map.items():
        if ext in joined:
            return fmt
    return None


def _classify_accession_category(accession: DataAccession, paper_type: str | None) -> str:
    repo = (accession.repository or "").strip().lower()
    desc = (accession.description or "").strip().lower()
    url = (accession.url or "").strip().lower()
    acc = (accession.accession_id or "").strip().lower()

    non_data_tokens = {
        "jupyter",
        "render",
        "covidence",
        "web of science",
        "acm digital library",
        "artificial intelligence act",
        "eu ai act",
        "nature scientific data",
        "springer",
    }
    hay = " ".join([repo, desc, url, acc])
    if any(token in hay for token in non_data_tokens):
        return "external_reference"

    dataset_repos = {
        "geo",
        "sra",
        "ena",
        "figshare",
        "fgshare",
        "zenodo",
        "dryad",
        "osf",
        "dataverse",
        "kaggle",
        "pangaea",
        "arrayexpress",
    }
    repo_hit = any(token in repo for token in dataset_repos) or any(
        token in url for token in dataset_repos
    )
    if repo_hit:
        if paper_type == "dataset_descriptor":
            return "primary_dataset"
        primary_tokens = {"this paper", "this study", "our dataset", "released dataset", "curated dataset"}
        if any(token in desc for token in primary_tokens):
            return "primary_dataset"
        return "supplementary_data"

    accession_prefixes = ("gse", "srp", "srr", "erp", "drp", "prjna", "ega", "pxd")
    if acc.startswith(accession_prefixes):
        return "supplementary_data"
    return "external_reference"


def _resource_type(name: str, url: str) -> str:
    text = f"{name} {url}".lower()
    if any(k in text for k in ("render", "dash", "plotly", "visualization", "sunburst")):
        return "visualization"
    if any(k in text for k in ("covidence", "jupyter", "github", "tool", "software")):
        return "tool"
    if any(k in text for k in ("oecd", "eu ai act", "standard", "regulation", "guideline")):
        return "standard"
    return "related_dataset"


def _extract_related_resources_from_text(paper_markdown: str) -> list[RelatedResource]:
    allow_domains = {
        "render.com",
        "covidence.org",
        "oecd.ai",
        "airc.nist.gov",
        "global-index.ai",
        "jupyter.org",
    }
    urls = re.findall(r"https?://[^\s)\]]+", paper_markdown, flags=re.IGNORECASE)
    out: list[RelatedResource] = []
    seen: set[str] = set()
    for url in urls:
        clean = _norm_url_ocr(url.strip().rstrip(".,;"))
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        if any(k in key for k in ("figshare", "fgshare", "zenodo", "gse", "sra", "10.6084/m9")):
            continue
        if any(k in key for k in ("doi.org", "arxiv.org")):
            continue
        name = re.sub(r"^https?://", "", clean).split("/")[0]
        domain = name.lower().replace("www.", "")
        if domain not in allow_domains:
            continue
        out.append(
            RelatedResource(
                name=name,
                url=clean,
                type=_resource_type(name, clean),
                description="Related resource mentioned in paper.",
            )
        )
    return out


def _dedupe_related_resources(resources: list[RelatedResource]) -> list[RelatedResource]:
    out: list[RelatedResource] = []
    seen: set[str] = set()
    for r in resources:
        if r.url:
            r.url = _norm_url_ocr(r.url)
        key = f"{(r.name or '').strip().lower()}::{(r.url or '').strip().lower()}::{(r.type or '').strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _sanitize_data_availability_payload(payload: dict) -> dict:
    accessions = payload.get("data_accessions")
    if not isinstance(accessions, list):
        accessions = []
    related = payload.get("related_resources")
    if not isinstance(related, list):
        related = []
    report = payload.get("data_availability")
    if not isinstance(report, dict):
        report = {
            "overall_status": "not_checked",
            "claimed_repositories": [],
            "verified_repositories": [],
            "discrepancies": [],
            "notes": "Repair parser fallback used.",
            "check_status": "failed",
        }
    report.setdefault("check_status", "ok")
    return {
        "data_accessions": accessions,
        "related_resources": related,
        "data_availability": report,
    }


def _normalize_overall_status(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in {"accessible", "partially_accessible", "unavailable", "not_checked"}:
        return raw
    if "partial" in raw:
        return "partially_accessible"
    if "unavailable" in raw:
        return "unavailable"
    if "accessible" in raw:
        return "accessible"
    return "not_checked"


async def _enrich_accession(accession: DataAccession) -> DataAccession:
    """Deterministically enrich repository accessions with live checks."""
    repo = _normalize_repo_name(accession)
    original_url = accession.url
    accession.url = _norm_url_ocr(accession.url)
    accession.accession_id = _norm_url_ocr(accession.accession_id) or accession.accession_id
    identifier = accession.url or accession.accession_id

    if accession.url:
        ping = await check_url_request(accession.url)
        accession.is_accessible = bool(ping.get("is_accessible"))
        if not accession.is_accessible:
            for candidate in _repair_url_candidates(original_url):
                if candidate == accession.url:
                    continue
                retry = await check_url_request(candidate)
                if bool(retry.get("is_accessible")):
                    accession.url = candidate
                    accession.is_accessible = True
                    accession.url_repaired = True
                    break
        elif accession.url != (original_url or "").strip():
            accession.url_repaired = True

    if "zenodo" in repo:
        info = await check_zenodo_record_request(identifier)
        if info.get("exists"):
            accession.is_accessible = True
            accession.file_count = int(info.get("file_count") or 0)
            accession.files_listed = list(info.get("files") or [])
            accession.total_size_bytes = int(info.get("total_size_bytes") or 0)
            sample_url = info.get("sample_file_url")
            accession.download_probe_url = str(sample_url) if sample_url else accession.url
            if sample_url and accession.total_size_bytes:
                probe = await estimate_download_time_request(str(sample_url))
                speed = probe.get("bytes_per_second")
                if isinstance(speed, (int, float)) and speed > 0:
                    accession.estimated_download_seconds = accession.total_size_bytes / float(speed)
    elif "figshare" in repo or "fgshare" in repo:
        info = await check_figshare_record_request(identifier)
        if info.get("exists"):
            accession.is_accessible = True
            accession.file_count = int(info.get("file_count") or 0)
            accession.files_listed = list(info.get("files") or [])
            accession.total_size_bytes = int(info.get("total_size_bytes") or 0)
            sample_url = info.get("sample_file_url")
            accession.download_probe_url = str(sample_url) if sample_url else accession.url
            if sample_url and accession.total_size_bytes:
                probe = await estimate_download_time_request(str(sample_url))
                speed = probe.get("bytes_per_second")
                if isinstance(speed, (int, float)) and speed > 0:
                    accession.estimated_download_seconds = accession.total_size_bytes / float(speed)
    elif repo == "geo" or accession.accession_id.upper().startswith("GSE"):
        info = await check_geo_accession_request(accession.accession_id)
        accession.is_accessible = bool(info.get("exists"))
        files = list(info.get("file_listing") or [])
        accession.file_count = len(files) if files else accession.file_count
        accession.files_listed = files or accession.files_listed
    elif repo == "sra" or accession.accession_id.upper().startswith(("SRP", "SRA", "SRR")):
        info = await check_sra_accession_request(accession.accession_id)
        accession.is_accessible = bool(info.get("exists"))
        files = list(info.get("file_listing") or [])
        accession.file_count = len(files) if files else accession.file_count
        accession.files_listed = files or accession.files_listed

    return accession


async def _enrich_data_accessions(accessions: list[DataAccession]) -> list[DataAccession]:
    enriched: list[DataAccession] = []
    for accession in accessions:
        try:
            enriched.append(await _enrich_accession(accession))
        except Exception as exc:  # noqa: BLE001
            log_event(
                "agent.data_availability.enrich_error",
                {"accession_id": accession.accession_id, "error": str(exc)},
            )
            enriched.append(accession)
    return enriched


def _extract_figshare_file_mentions(paper_markdown: str) -> list[str]:
    patterns = [
        r"\bREADME(?:\.md)?\b",
        r"\b[^ \n\t]+\.xlsx\b",
        r"\b[^ \n\t]+\.csv\b",
        r"\b[^ \n\t]+\.tsv\b",
        r"\b[^ \n\t]+\.ipynb\b",
        r"\b[^ \n\t]+\.py\b",
        r"\b[^ \n\t]+\.md\b",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, paper_markdown, flags=re.IGNORECASE):
            token = re.sub(r"[),.;]+$", "", str(match).strip())
            key = token.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            found.append(token)
    return found[:80]


def _extract_candidate_accessions_from_text(
    paper_markdown: str,
    paper_type: str | None,
) -> list[DataAccession]:
    repo_patterns = {
        "figshare": r"https?://[^\s)]+(?:figshare|fgshare)[^\s)]*",
        "zenodo": r"https?://[^\s)]+zenodo[^\s)]*",
        "geo": r"\bGSE[0-9]+\b",
        "sra": r"\bSRP[0-9]+\b|\bSRR[0-9]+\b",
        "figshare_doi": r"\b10\.6084/m9\.figshare\.[0-9]+\b|\b10\.6084/m9\.fgshare\.[0-9]+\b",
    }
    found: list[DataAccession] = []
    category = "primary_dataset" if paper_type == "dataset_descriptor" else "supplementary_data"
    for repo, pattern in repo_patterns.items():
        for idx, match in enumerate(re.findall(pattern, paper_markdown, flags=re.IGNORECASE), start=1):
            token = _norm_url_ocr(str(match).strip()) or str(match).strip()
            accession_id = token
            url = token if token.startswith("http") else None
            if repo == "figshare_doi" and not url:
                url = f"https://doi.org/{token}"
            if repo == "geo" and not url:
                accession_id = token.upper()
            if repo == "sra" and not url:
                accession_id = token.upper()
            repo_name = "Figshare" if repo.startswith("figshare") else ("GEO" if repo == "geo" else ("SRA" if repo == "sra" else repo.capitalize()))
            found.append(
                DataAccession(
                    accession_id=accession_id,
                    category=category,
                    repository=repo_name,
                    url=url,
                    description="Recovered from dataset-like pattern in paper text.",
                )
            )
    deduped: list[DataAccession] = []
    seen: set[str] = set()
    for item in found:
        acc_key = item.accession_id.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
        key = f"{item.repository.lower()}::{acc_key}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


async def run_data_availability_agent(
    paper_markdown: str,
    paper_type: str | None = None,
    guidance: str | None = None,
) -> DataAvailabilityOutput:
    log_event("agent.data_availability.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    if paper_type:
        agent_input = f"[PAPER_TYPE]\n{paper_type}\n\n{agent_input}"
    try:
        result = await run_with_rate_limit_retry(
            lambda: Runner.run(data_availability_agent, input=agent_input)
        )
        output = result.final_output
        if not isinstance(output, DataAvailabilityOutput):
            output = DataAvailabilityOutput.model_validate(output)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.data_availability.retry_on_parse_error", {"error": str(exc)})
        repair_input = (
            f"{agent_input}\n\n"
            "[FORMAT_FIX]\n"
            "Return JSON object with keys exactly:\n"
            "data_accessions, related_resources, data_availability.\n"
            "Return populated values only. Do NOT return schema objects.\n"
        )
        repair_result = await run_with_rate_limit_retry(
            lambda: Runner.run(data_availability_repair_agent, input=repair_input)
        )
        raw = repair_result.final_output
        if isinstance(raw, DataAvailabilityOutput):
            output = raw
        elif isinstance(raw, dict):
            output = DataAvailabilityOutput.model_validate(_sanitize_data_availability_payload(raw))
        else:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                raise ValueError("data_availability_repair_agent returned non-object JSON")
            output = DataAvailabilityOutput.model_validate(_sanitize_data_availability_payload(parsed))
    enriched = await _enrich_data_accessions(output.data_accessions)
    kept: list[DataAccession] = []
    related_from_filtered: list[RelatedResource] = []
    dropped = 0
    for accession in enriched:
        accession.category = _classify_accession_category(accession, paper_type)
        if not accession.data_format:
            accession.data_format = _infer_data_format(accession)
        if accession.category == "external_reference":
            dropped += 1
            related_from_filtered.append(
                RelatedResource(
                    name=accession.repository or accession.accession_id,
                    url=accession.url,
                    type=_resource_type(accession.repository or accession.accession_id, accession.url or ""),
                    description=accession.description,
                )
            )
            continue
        kept.append(accession)
    output.data_accessions = kept
    if not output.data_accessions:
        recovered = _extract_candidate_accessions_from_text(paper_markdown, paper_type)
        if recovered:
            output.data_accessions = await _enrich_data_accessions(recovered)
            output.data_availability.discrepancies.append(
                "Recovered dataset-like accessions from paper text fallback parser."
            )
    figshare_mentions = _extract_figshare_file_mentions(paper_markdown)
    if figshare_mentions:
        for accession in output.data_accessions:
            repo_name = (accession.repository or "").lower()
            if "figshare" in repo_name or "fgshare" in repo_name:
                files = list(accession.files_listed or [])
                seen = {x.lower() for x in files}
                for mention in figshare_mentions:
                    if mention.lower() in seen:
                        continue
                    files.append(mention)
                    seen.add(mention.lower())
                accession.files_listed = files[:120]
                accession.file_count = max(accession.file_count or 0, len(accession.files_listed))
                if accession.description and "zip" not in accession.description.lower():
                    accession.description = f"{accession.description} Repository package may contain multiple files inside zip/archive."
    if not output.related_resources:
        output.related_resources = _extract_related_resources_from_text(paper_markdown)
    if related_from_filtered:
        output.related_resources.extend(related_from_filtered)
    output.related_resources = _sanitize_reference_noise(output.related_resources, paper_markdown)
    output.related_resources = _dedupe_related_resources(output.related_resources)
    claimed = sorted({(x.repository or "").strip() for x in output.data_accessions if (x.repository or "").strip()})
    verified = sorted(
        {
            (x.repository or "").strip()
            for x in output.data_accessions
            if x.is_accessible is True and (x.repository or "").strip()
        }
    )
    if claimed:
        output.data_availability.claimed_repositories = claimed
    output.data_availability.verified_repositories = verified

    if output.data_accessions:
        statuses = [x.is_accessible for x in output.data_accessions]
        if all(s is True for s in statuses):
            output.data_availability.overall_status = "accessible"
            output.data_availability.check_status = "ok"
        elif any(s is True for s in statuses):
            output.data_availability.overall_status = "partially_accessible"
            output.data_availability.check_status = "partial"
        elif all(s is False for s in statuses):
            output.data_availability.overall_status = "unavailable"
            output.data_availability.check_status = "failed"
        else:
            output.data_availability.overall_status = _normalize_overall_status(output.data_availability.overall_status)
            output.data_availability.check_status = "partial"
    else:
        output.data_availability.overall_status = _normalize_overall_status(output.data_availability.overall_status)
        notes_text = (output.data_availability.notes or "").lower()
        if any(k in notes_text for k in ("could not", "failed", "error", "timeout", "unverified")):
            output.data_availability.check_status = "failed"
        else:
            output.data_availability.check_status = "ok"
    if dropped:
        output.data_availability.discrepancies.append(
            f"Filtered {dropped} external reference links from data_accessions."
        )
    discrepancy_text = " ".join(output.data_availability.discrepancies).lower()
    unresolved_markers = ("could not", "failed", "dns", "unverified", "timeout", "error")
    if any(marker in discrepancy_text for marker in unresolved_markers):
        if output.data_availability.overall_status == "accessible":
            output.data_availability.overall_status = "partially_accessible"
        if output.data_availability.check_status == "ok":
            output.data_availability.check_status = "partial"
    if output.data_availability.check_status == "failed" and output.data_availability.overall_status == "accessible":
        output.data_availability.overall_status = "partially_accessible"
    log_event(
        "agent.data_availability.end",
        {
            "data_accessions": len(output.data_accessions),
            "overall_status": output.data_availability.overall_status,
        },
    )
    return output


async def fallback_data_availability_from_text(
    paper_markdown: str,
    paper_type: str | None = None,
    reason: str | None = None,
) -> DataAvailabilityOutput:
    recovered = _extract_candidate_accessions_from_text(paper_markdown, paper_type)
    enriched = await _enrich_data_accessions(recovered)
    any_true = any(x.is_accessible is True for x in enriched)
    any_false = any(x.is_accessible is False for x in enriched)
    if any_true and any_false:
        status = "partially_accessible"
    elif any_true:
        status = "partially_accessible"
    elif any_false:
        status = "unavailable"
    else:
        status = "not_checked"
    discrepancies: list[str] = []
    if reason:
        discrepancies.append(f"data_availability_agent_error: {reason}")
    if recovered:
        discrepancies.append("Recovered dataset-like accessions from fallback text parser.")
    return DataAvailabilityOutput(
        data_accessions=enriched,
        related_resources=_extract_related_resources_from_text(paper_markdown),
        data_availability=DataAvailabilityReport(
            overall_status=status,
            claimed_repositories=sorted({x.repository for x in enriched}),
            verified_repositories=sorted({x.repository for x in enriched if x.is_accessible is True}),
            discrepancies=discrepancies,
            notes="Fallback data-availability parser used.",
            check_status="failed",
        ),
    )
