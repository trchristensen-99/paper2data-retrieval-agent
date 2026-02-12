from __future__ import annotations

import re

from pydantic import BaseModel, Field
from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import DataAccession, DataAvailabilityReport
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


def _normalize_repo_name(accession: DataAccession) -> str:
    return accession.repository.strip().lower()


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


async def _enrich_accession(accession: DataAccession) -> DataAccession:
    """Deterministically enrich repository accessions with live checks."""
    repo = _normalize_repo_name(accession)
    identifier = accession.url or accession.accession_id

    if accession.url:
        ping = await check_url_request(accession.url)
        accession.is_accessible = bool(ping.get("is_accessible"))

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
    elif "figshare" in repo:
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


def _extract_candidate_accessions_from_text(
    paper_markdown: str,
    paper_type: str | None,
) -> list[DataAccession]:
    repo_patterns = {
        "figshare": r"https?://[^\s)]+(?:figshare|fgshare)[^\s)]*",
        "zenodo": r"https?://[^\s)]+zenodo[^\s)]*",
        "geo": r"\bGSE[0-9]+\b",
        "sra": r"\bSRP[0-9]+\b|\bSRR[0-9]+\b",
        "figshare_doi": r"\b10\.6084/m9\.fgshare\.[0-9]+\b",
    }
    found: list[DataAccession] = []
    category = "primary_dataset" if paper_type == "dataset_descriptor" else "supplementary_data"
    for repo, pattern in repo_patterns.items():
        for idx, match in enumerate(re.findall(pattern, paper_markdown, flags=re.IGNORECASE), start=1):
            token = str(match).strip()
            accession_id = token
            url = token if token.startswith("http") else None
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
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(data_availability_agent, input=agent_input)
    )
    output = result.final_output
    if not isinstance(output, DataAvailabilityOutput):
        output = DataAvailabilityOutput.model_validate(output)
    enriched = await _enrich_data_accessions(output.data_accessions)
    kept: list[DataAccession] = []
    dropped = 0
    for accession in enriched:
        accession.category = _classify_accession_category(accession, paper_type)
        if not accession.data_format:
            accession.data_format = _infer_data_format(accession)
        if accession.category == "external_reference":
            dropped += 1
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
    if dropped:
        output.data_availability.discrepancies.append(
            f"Filtered {dropped} external reference links from data_accessions."
        )
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
    status = "accessible" if any(x.is_accessible for x in enriched) else "not_checked"
    discrepancies: list[str] = []
    if reason:
        discrepancies.append(f"data_availability_agent_error: {reason}")
    if recovered:
        discrepancies.append("Recovered dataset-like accessions from fallback text parser.")
    return DataAvailabilityOutput(
        data_accessions=enriched,
        data_availability=DataAvailabilityReport(
            overall_status=status,
            claimed_repositories=sorted({x.repository for x in enriched}),
            verified_repositories=sorted({x.repository for x in enriched if x.is_accessible}),
            discrepancies=discrepancies,
            notes="Fallback data-availability parser used.",
        ),
    )
