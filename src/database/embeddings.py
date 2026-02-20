from __future__ import annotations

import json
import math
import os
from typing import Any

from openai import OpenAI


EMBED_MODEL = os.getenv("P2D_EMBED_MODEL", "text-embedding-3-small")


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=text)
    vec = resp.data[0].embedding
    return [float(x) for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def embedding_payload_for_record(record_json: dict[str, Any]) -> str:
    metadata = record_json.get("metadata", {})
    methods = record_json.get("methods", {})
    results = record_json.get("results", {})
    parts = [
        str(metadata.get("title") or ""),
        " ".join(str(x) for x in (metadata.get("keywords") or [])),
        str(metadata.get("paper_type") or ""),
        str(metadata.get("paper_archetype") or ""),
        str(methods.get("experimental_design") or ""),
        " ".join(str(x) for x in (methods.get("assay_types") or [])),
        " ".join(str(x) for x in (methods.get("organisms") or [])),
        " ".join(str(x) for x in (results.get("synthesized_claims") or [])),
    ]
    findings = results.get("experimental_findings") or []
    for finding in findings[:12]:
        if not isinstance(finding, dict):
            continue
        parts.append(
            " ".join(
                [
                    str(finding.get("metric") or ""),
                    str(finding.get("value") or ""),
                    str(finding.get("context") or ""),
                ]
            )
        )
    return "\n".join([p for p in parts if p.strip()])


def encode_embedding_json(vec: list[float]) -> str:
    return json.dumps(vec, separators=(",", ":"))


def decode_embedding_json(payload: str | None) -> list[float]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            return [float(x) for x in data]
    except Exception:  # noqa: BLE001
        return []
    return []

