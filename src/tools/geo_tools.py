from __future__ import annotations

import re
from typing import Any

import httpx
from agents import function_tool

from src.utils.logging import log_event

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _extract_file_candidates(text: str) -> list[str]:
    candidates = re.findall(r"\b\S+\.(?:fastq(?:\.gz)?|bam|csv|tsv|txt|bed|bigwig|wig)\b", text, flags=re.IGNORECASE)
    return sorted(set(candidates))


@function_tool
async def check_geo_accession(accession: str) -> dict:
    """Verify a GEO accession and return a lightweight summary from NCBI E-utilities."""
    result: dict[str, Any] = {
        "accession": accession,
        "exists": False,
        "sample_count": None,
        "platform": None,
        "file_listing": [],
        "error": None,
    }

    try:
        params = {
            "db": "gds",
            "term": accession,
            "retmode": "json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                log_event("tool.check_geo_accession", {**result})
                return result

            summary = await client.get(
                f"{EUTILS_BASE}/esummary.fcgi",
                params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
            )
            summary.raise_for_status()
            data = summary.json().get("result", {})

        sample_count = 0
        platform = None
        files: list[str] = []
        for uid in ids:
            entry = data.get(uid, {})
            result["exists"] = True
            sample_count += int(entry.get("n_samples", 0) or 0)
            if not platform and entry.get("gpl"):
                platform = str(entry["gpl"])
            files.extend(_extract_file_candidates(str(entry)))

        result["sample_count"] = sample_count
        result["platform"] = platform
        result["file_listing"] = sorted(set(files))
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.check_geo_accession", {**result})
    return result


@function_tool
async def check_sra_accession(accession: str) -> dict:
    """Verify an SRA accession using NCBI E-utilities and return lightweight metadata."""
    result: dict[str, Any] = {
        "accession": accession,
        "exists": False,
        "run_count": None,
        "file_listing": [],
        "error": None,
    }

    try:
        params = {
            "db": "sra",
            "term": accession,
            "retmode": "json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                log_event("tool.check_sra_accession", {**result})
                return result

            summary = await client.get(
                f"{EUTILS_BASE}/esummary.fcgi",
                params={"db": "sra", "id": ",".join(ids), "retmode": "json"},
            )
            summary.raise_for_status()
            data = summary.json().get("result", {})

        run_count = 0
        files: list[str] = []
        for uid in ids:
            entry = data.get(uid, {})
            result["exists"] = True
            runs_value = entry.get("runs")
            if isinstance(runs_value, str):
                run_count += max(1, runs_value.count("Run acc=") )
            files.extend(_extract_file_candidates(str(entry)))

        result["run_count"] = run_count
        result["file_listing"] = sorted(set(files))
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.check_sra_accession", {**result})
    return result
