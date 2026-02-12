from __future__ import annotations

import httpx
from agents import function_tool

from src.utils.logging import log_event


@function_tool
async def search_crossref_by_title(title: str) -> dict:
    """Search Crossref for DOI and bibliographic fields using paper title."""
    result = {
        "title_query": title,
        "doi": None,
        "journal": None,
        "publication_date": None,
        "score": None,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query.title": title, "rows": 1},
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
            if items:
                top = items[0]
                result["doi"] = top.get("DOI")
                container = top.get("container-title") or []
                if container:
                    result["journal"] = container[0]
                pub_parts = (
                    top.get("published-print", {}).get("date-parts")
                    or top.get("published-online", {}).get("date-parts")
                    or []
                )
                if pub_parts and pub_parts[0]:
                    result["publication_date"] = "-".join(str(x) for x in pub_parts[0])
                result["score"] = top.get("score")
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.search_crossref_by_title", result)
    return result


@function_tool
async def search_pubmed(title_or_doi: str) -> dict:
    """Search PubMed via NCBI E-utilities and return PMID if present."""
    result = {
        "query": title_or_doi,
        "pmid": None,
        "match_count": 0,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": title_or_doi, "retmode": "json", "retmax": 1},
            )
            search.raise_for_status()
            payload = search.json().get("esearchresult", {})
            ids = payload.get("idlist", [])
            result["match_count"] = int(payload.get("count", 0) or 0)
            if ids:
                result["pmid"] = ids[0]
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.search_pubmed", result)
    return result
