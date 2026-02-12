from __future__ import annotations

from dataclasses import dataclass

from agents import Agent

from src.agents.data_availability import (
    DataAvailabilityOutput,
    data_availability_agent,
    run_data_availability_agent,
)
from src.agents.metadata import metadata_agent, run_metadata_agent
from src.agents.methods import methods_agent, run_methods_agent
from src.agents.results import results_agent, run_results_agent
from src.agents.synthesis import fallback_synthesis, run_synthesis_agent, synthesis_agent
from src.schemas.models import PaperRecord, SynthesisInput, SynthesisOutput
from src.utils.config import MODELS
from src.utils.logging import log_event, reset_events


manager_agent = Agent(
    name="manager_agent",
    model=MODELS.manager,
    instructions=(
        "You orchestrate handoffs across metadata -> methods -> results -> "
        "data_availability -> synthesis."
    ),
    handoffs=[metadata_agent, methods_agent, results_agent, data_availability_agent, synthesis_agent],
)


@dataclass
class PipelineArtifacts:
    record: PaperRecord
    retrieval_report_markdown: str
    retrieval_log_markdown: str


async def run_pipeline(paper_markdown: str) -> PipelineArtifacts:
    reset_events()
    log_event("pipeline.start", {"chars": len(paper_markdown)})

    metadata = await run_metadata_agent(paper_markdown)
    methods = await run_methods_agent(paper_markdown)
    results = await run_results_agent(paper_markdown)
    data_availability: DataAvailabilityOutput = await run_data_availability_agent(paper_markdown)

    synthesis_input = SynthesisInput(
        metadata=metadata,
        methods=methods,
        results=results,
        data_accessions=data_availability.data_accessions,
        data_availability=data_availability.data_availability,
    )

    try:
        synthesis: SynthesisOutput = await run_synthesis_agent(synthesis_input)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.synthesis.error", {"error": str(exc)})
        synthesis = fallback_synthesis(synthesis_input)

    log_event("pipeline.end", {"confidence": synthesis.record.extraction_confidence})
    return PipelineArtifacts(
        record=synthesis.record,
        retrieval_report_markdown=synthesis.retrieval_report_markdown,
        retrieval_log_markdown=synthesis.retrieval_log_markdown,
    )
