from __future__ import annotations

import os
import socket
from typing import Any

from src.utils.http_client import make_sync_client


def check_openai_dns(host: str = "api.openai.com") -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return True, f"DNS resolution succeeded for {host}"
    except Exception as exc:  # noqa: BLE001
        return False, f"DNS resolution failed for {host}: {exc}"


def check_external_service_access(timeout_seconds: float = 12.0) -> tuple[bool, str, list[dict[str, Any]]]:
    """
    Probe core external services used by tools.
    Returns: (all_ok, summary_message, per_service_results)
    """
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    endpoints: list[dict[str, Any]] = [
        {
            "name": "openai_models",
            "method": "GET",
            "url": "https://api.openai.com/v1/models",
            "headers": {"Authorization": f"Bearer {key}"} if key else {},
            "ok_statuses": {200, 401},
        },
        {
            "name": "ncbi_eutils",
            "method": "GET",
            "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term=GSE1&retmode=json",
            "headers": {},
            "ok_statuses": {200},
        },
        {
            "name": "crossref_api",
            "method": "GET",
            "url": "https://api.crossref.org/works?rows=1",
            "headers": {},
            "ok_statuses": {200},
        },
        {
            "name": "figshare_api",
            "method": "GET",
            "url": "https://api.figshare.com/v2/articles?page_size=1",
            "headers": {},
            "ok_statuses": {200},
        },
        {
            "name": "zenodo_api",
            "method": "GET",
            "url": "https://zenodo.org/api/records?q=dataset&size=1",
            "headers": {},
            "ok_statuses": {200},
        },
        {
            "name": "doi_resolver",
            "method": "GET",
            "url": "https://doi.org/10.1038/s41597-025-06021-5",
            "headers": {},
            "ok_statuses": {200},
        },
    ]

    checks: list[dict[str, Any]] = []
    with make_sync_client(timeout_seconds=timeout_seconds, follow_redirects=True) as client:
        for endpoint in endpoints:
            entry = {
                "name": endpoint["name"],
                "url": endpoint["url"],
                "ok": False,
                "status_code": None,
                "error": None,
            }
            try:
                resp = client.request(
                    endpoint["method"],
                    endpoint["url"],
                    headers=endpoint.get("headers") or None,
                )
                entry["status_code"] = resp.status_code
                entry["ok"] = resp.status_code in endpoint["ok_statuses"]
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)
            checks.append(entry)

    failed = [c["name"] for c in checks if not c["ok"]]
    if failed:
        return False, f"External service checks failed: {', '.join(failed)}", checks
    return True, "External service checks passed", checks
