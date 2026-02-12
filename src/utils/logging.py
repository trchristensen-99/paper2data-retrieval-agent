from __future__ import annotations

from datetime import datetime
from typing import Any

RETRIEVAL_EVENTS: list[dict[str, Any]] = []


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    RETRIEVAL_EVENTS.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
    )


def reset_events() -> None:
    RETRIEVAL_EVENTS.clear()


def events_as_markdown() -> str:
    lines = ["# Retrieval Log", ""]
    for event in RETRIEVAL_EVENTS:
        lines.append(
            f"- [{event['timestamp']}] **{event['event_type']}**: {event['payload']}"
        )
    if len(lines) == 2:
        lines.append("- No retrieval events captured.")
    return "\n".join(lines)
