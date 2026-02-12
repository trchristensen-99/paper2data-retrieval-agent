from __future__ import annotations

from pydantic import BaseModel, Field
from agents import Agent, Runner

from src.schemas.models import DataAccession, DataAvailabilityReport
from src.tools.file_tools import list_ftp_files
from src.tools.geo_tools import check_geo_accession, check_sra_accession
from src.tools.http_tools import check_url
from src.utils.config import MODELS
from src.utils.logging import log_event


class DataAvailabilityOutput(BaseModel):
    data_accessions: list[DataAccession] = Field(default_factory=list)
    data_availability: DataAvailabilityReport


DATA_AVAILABILITY_PROMPT = """Identify all data accessions and repository links in the paper.
For each relevant accession/URL, use tools to verify accessibility:
- check_url for web links
- check_geo_accession for GEO IDs
- check_sra_accession for SRA IDs
- list_ftp_files for FTP resources

Compare claimed availability against verified availability and record discrepancies.
"""


data_availability_agent = Agent(
    name="data_availability_agent",
    model=MODELS.data_availability,
    instructions=DATA_AVAILABILITY_PROMPT,
    tools=[check_url, check_geo_accession, check_sra_accession, list_ftp_files],
    output_type=DataAvailabilityOutput,
)


async def run_data_availability_agent(paper_markdown: str) -> DataAvailabilityOutput:
    log_event("agent.data_availability.start", {"chars": len(paper_markdown)})
    result = await Runner.run(data_availability_agent, input=paper_markdown)
    output = result.final_output
    if not isinstance(output, DataAvailabilityOutput):
        output = DataAvailabilityOutput.model_validate(output)
    log_event(
        "agent.data_availability.end",
        {
            "data_accessions": len(output.data_accessions),
            "overall_status": output.data_availability.overall_status,
        },
    )
    return output
