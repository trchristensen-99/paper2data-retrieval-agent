from __future__ import annotations

import re

_MAP = {
    "human": ("Homo sapiens", "human"),
    "homo sapiens": ("Homo sapiens", "human"),
    "h. sapiens": ("Homo sapiens", "human"),
    "mouse": ("Mus musculus", "mouse"),
    "mus musculus": ("Mus musculus", "mouse"),
    "m. musculus": ("Mus musculus", "mouse"),
    "rat": ("Rattus norvegicus", "rat"),
    "rattus norvegicus": ("Rattus norvegicus", "rat"),
    "zebrafish": ("Danio rerio", "zebrafish"),
    "danio rerio": ("Danio rerio", "zebrafish"),
    "fruit fly": ("Drosophila melanogaster", "fruit fly"),
    "drosophila melanogaster": ("Drosophila melanogaster", "fruit fly"),
    "yeast": ("Saccharomyces cerevisiae", "baker's yeast"),
    "saccharomyces cerevisiae": ("Saccharomyces cerevisiae", "baker's yeast"),
    "arabidopsis": ("Arabidopsis thaliana", "thale cress"),
    "arabidopsis thaliana": ("Arabidopsis thaliana", "thale cress"),
    "e. coli": ("Escherichia coli", "E. coli"),
    "escherichia coli": ("Escherichia coli", "E. coli"),
}


def _norm(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip().lower())


def _fmt(latin: str, common: str | None = None) -> str:
    if common and common.strip():
        return f"{latin} [common: {common}]"
    return latin


def normalize_organism_entries(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = raw.strip()
        if not text:
            continue
        k = _norm(text)
        latin, common = _MAP.get(k, (text, None))
        formatted = _fmt(latin, common)
        dedupe_key = _norm(latin)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(formatted)

    if len(out) > 20:
        total = len(out)
        out = out[:20] + [f"MULTI_SPECIES(total={total})"]
    return out
