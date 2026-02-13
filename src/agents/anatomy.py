from __future__ import annotations

import json
import re

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import PaperAnatomyOutput
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry


ANATOMY_PROMPT = """Extract a structural map of the paper for downstream agents.
Return only concrete extracted values:
- sections found
- table labels/titles
- figure labels/titles
- all URLs
- accession-like identifiers (GSE/SRP/SRR/DOI-like data ids)
- PRISMA/scoping-flow counts when present

Do not interpret findings; this is structure-only context.
"""


anatomy_agent = Agent(
    name="anatomy_agent",
    model=MODELS.anatomy,
    instructions=ANATOMY_PROMPT,
    output_type=AgentOutputSchema(PaperAnatomyOutput, strict_json_schema=False),
)

anatomy_repair_agent = Agent(
    name="anatomy_repair_agent",
    model=MODELS.anatomy,
    instructions=(
        ANATOMY_PROMPT
        + "\nReturn only populated JSON values for: "
        + "sections, tables, figures, urls, accession_candidates, prisma_flow, notes."
        + "\n`prisma_flow` must be a flat object mapping string->integer counts."
        + "\nDo NOT return nested schema-like objects."
    ),
)


def _sanitize_prisma_flow(value) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, int):
                out[str(key)] = item
            elif isinstance(item, str):
                cleaned = re.sub(r"[^0-9]", "", item)
                if cleaned:
                    out[str(key)] = int(cleaned)
            elif isinstance(item, dict):
                nested = _sanitize_prisma_flow(item)
                for n_key, n_val in nested.items():
                    out[f"{key}_{n_key}"] = n_val
    return out


def _sanitize_anatomy_payload(payload: dict) -> dict:
    return {
        "sections": payload.get("sections") if isinstance(payload.get("sections"), list) else [],
        "tables": payload.get("tables") if isinstance(payload.get("tables"), list) else [],
        "figures": payload.get("figures") if isinstance(payload.get("figures"), list) else [],
        "urls": payload.get("urls") if isinstance(payload.get("urls"), list) else [],
        "accession_candidates": payload.get("accession_candidates")
        if isinstance(payload.get("accession_candidates"), list)
        else [],
        "prisma_flow": _sanitize_prisma_flow(payload.get("prisma_flow")),
        "notes": str(payload.get("notes") or ""),
    }


async def run_anatomy_agent(paper_markdown: str) -> PaperAnatomyOutput:
    log_event("agent.anatomy.start", {"chars": len(paper_markdown)})
    try:
        result = await run_with_rate_limit_retry(lambda: Runner.run(anatomy_agent, input=paper_markdown))
        output = result.final_output
        if not isinstance(output, PaperAnatomyOutput):
            output = PaperAnatomyOutput.model_validate(_sanitize_anatomy_payload(output))
    except Exception as exc:  # noqa: BLE001
        log_event("agent.anatomy.retry_on_parse_error", {"error": str(exc)})
        repair_input = (
            f"{paper_markdown}\n\n"
            "[FORMAT_FIX]\n"
            "Return JSON with keys: sections, tables, figures, urls, accession_candidates, prisma_flow, notes.\n"
            "Use flat string->integer pairs for prisma_flow.\n"
        )
        try:
            repair_result = await run_with_rate_limit_retry(
                lambda: Runner.run(anatomy_repair_agent, input=repair_input)
            )
            raw = repair_result.final_output
            if isinstance(raw, PaperAnatomyOutput):
                output = raw
            elif isinstance(raw, dict):
                output = PaperAnatomyOutput.model_validate(_sanitize_anatomy_payload(raw))
            else:
                output = PaperAnatomyOutput()
        except Exception as repair_exc:  # noqa: BLE001
            log_event("agent.anatomy.repair_failed", {"error": str(repair_exc)})
            output = PaperAnatomyOutput()
    # Deterministic backstops for common high-value structure signals.
    if not output.urls:
        output.urls = re.findall(r"https?://[^\s)\]]+", paper_markdown, flags=re.IGNORECASE)[:300]
    if not output.tables:
        output.tables = re.findall(r"\bTable\s+\d+\b", paper_markdown, flags=re.IGNORECASE)[:50]
    if not output.figures:
        output.figures = re.findall(r"\bFigure\s+\d+\b", paper_markdown, flags=re.IGNORECASE)[:50]

    if not output.prisma_flow:
        flow: dict[str, int] = {}
        lowered = paper_markdown.lower()
        pattern_map = {
            "duplicates_removed": r"\b([0-9][0-9,]*)\s+duplicates?\s+removed\b",
            "screened": r"\b([0-9][0-9,]*)\s+(?:records?\s+)?screened\b",
            "full_text_review": r"\b([0-9][0-9,]*)\s+full[\-\s]?text(?:\s+review)?\b",
            "included": r"\b([0-9][0-9,]*)\s+(?:papers?|studies|records?)\s+included\b",
        }
        for key, pattern in pattern_map.items():
            m = re.search(pattern, lowered)
            if m:
                flow[key] = int(re.sub(r"[^0-9]", "", m.group(1)))
        output.prisma_flow = flow

    log_event("agent.anatomy.end", json.loads(output.model_dump_json()))
    return output
