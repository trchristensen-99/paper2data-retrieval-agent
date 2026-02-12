from __future__ import annotations

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


async def run_data_availability_agent(
    paper_markdown: str, guidance: str | None = None
) -> DataAvailabilityOutput:
    log_event("agent.data_availability.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(data_availability_agent, input=agent_input)
    )
    output = result.final_output
    if not isinstance(output, DataAvailabilityOutput):
        output = DataAvailabilityOutput.model_validate(output)
    output.data_accessions = await _enrich_data_accessions(output.data_accessions)
    log_event(
        "agent.data_availability.end",
        {
            "data_accessions": len(output.data_accessions),
            "overall_status": output.data_availability.overall_status,
        },
    )
    return output
