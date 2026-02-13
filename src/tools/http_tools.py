from __future__ import annotations

from agents import function_tool

from src.utils.logging import log_event
from src.utils.http_client import make_async_client


async def check_url_request(url: str) -> dict:
    """Check whether an HTTP URL is reachable via HEAD (fallback to GET)."""
    result: dict[str, object] = {
        "url": url,
        "status_code": None,
        "is_accessible": False,
        "error": None,
    }
    try:
        async with make_async_client(timeout_seconds=15.0, follow_redirects=True) as client:
            response = await client.head(url)
            if response.status_code >= 400:
                response = await client.get(url)
            result["status_code"] = response.status_code
            result["is_accessible"] = response.status_code < 400
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.check_url", result)
    return result


@function_tool
async def check_url(url: str) -> dict:
    return await check_url_request(url)
