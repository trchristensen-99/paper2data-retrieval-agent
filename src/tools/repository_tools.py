from __future__ import annotations

import re
from time import perf_counter
from typing import Any

import httpx
from agents import function_tool

from src.utils.logging import log_event


ZENODO_ID_PATTERN = re.compile(r"zenodo\.(\d+)", re.IGNORECASE)
FIGSHARE_ID_PATTERN = re.compile(r"figshare\.(\d+)", re.IGNORECASE)
URL_NUMERIC_TAIL_PATTERN = re.compile(r"/(\d+)(?:\?.*)?$")


def _extract_zenodo_id(value: str) -> str | None:
    match = ZENODO_ID_PATTERN.search(value)
    if match:
        return match.group(1)
    match = re.search(r"/records/(\d+)", value)
    return match.group(1) if match else None


def _extract_figshare_id(value: str) -> str | None:
    match = FIGSHARE_ID_PATTERN.search(value)
    if match:
        return match.group(1)
    match = URL_NUMERIC_TAIL_PATTERN.search(value)
    return match.group(1) if match else None


async def check_zenodo_record_request(identifier: str) -> dict:
    """Validate a Zenodo DOI/URL and list available files and total size."""
    result: dict[str, Any] = {
        "identifier": identifier,
        "record_id": None,
        "exists": False,
        "title": None,
        "file_count": None,
        "total_size_bytes": None,
        "files": [],
        "sample_file_url": None,
        "error": None,
    }

    record_id = _extract_zenodo_id(identifier)
    result["record_id"] = record_id

    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            if not record_id:
                # Attempt DOI resolution to discover final Zenodo URL.
                resolved = await client.get(identifier)
                resolved.raise_for_status()
                record_id = _extract_zenodo_id(str(resolved.url))
                result["record_id"] = record_id
            if not record_id:
                raise ValueError(f"Unable to extract Zenodo record id from {identifier}")

            resp = await client.get(f"https://zenodo.org/api/records/{record_id}")
            resp.raise_for_status()
            payload = resp.json()

        files = payload.get("files", []) or []
        total_size = sum(int(f.get("size", 0) or 0) for f in files)
        file_names = [str(f.get("key") or f.get("filename") or "") for f in files if f]
        sample_url = None
        if files:
            links = files[0].get("links", {}) if isinstance(files[0], dict) else {}
            sample_url = links.get("self") or links.get("download")

        result.update(
            {
                "exists": True,
                "title": payload.get("metadata", {}).get("title"),
                "file_count": len(files),
                "total_size_bytes": total_size,
                "files": file_names[:200],
                "sample_file_url": sample_url,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.check_zenodo_record", result)
    return result


async def check_figshare_record_request(identifier: str) -> dict:
    """Validate a Figshare DOI/URL and list available files and total size."""
    result: dict[str, Any] = {
        "identifier": identifier,
        "article_id": None,
        "exists": False,
        "title": None,
        "file_count": None,
        "total_size_bytes": None,
        "files": [],
        "sample_file_url": None,
        "error": None,
    }

    article_id = _extract_figshare_id(identifier)
    result["article_id"] = article_id

    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            if not article_id:
                resolved = await client.get(identifier)
                resolved.raise_for_status()
                article_id = _extract_figshare_id(str(resolved.url))
                result["article_id"] = article_id
            if not article_id:
                raise ValueError(f"Unable to extract Figshare article id from {identifier}")

            resp = await client.get(f"https://api.figshare.com/v2/articles/{article_id}")
            resp.raise_for_status()
            payload = resp.json()

        files = payload.get("files", []) or []
        total_size = sum(int(f.get("size", 0) or 0) for f in files)
        file_names = [str(f.get("name") or "") for f in files if f]
        sample_url = None
        if files:
            sample_url = files[0].get("download_url")

        result.update(
            {
                "exists": True,
                "title": payload.get("title"),
                "file_count": len(files),
                "total_size_bytes": total_size,
                "files": file_names[:200],
                "sample_file_url": sample_url,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.check_figshare_record", result)
    return result


async def estimate_download_time_request(url: str, sample_bytes: int = 1024 * 1024) -> dict:
    """Estimate transfer speed and download time from a byte-range sample request."""
    result: dict[str, Any] = {
        "url": url,
        "sample_bytes": sample_bytes,
        "bytes_read": 0,
        "seconds": None,
        "bytes_per_second": None,
        "error": None,
    }

    headers = {"Range": f"bytes=0-{max(sample_bytes - 1, 1023)}"}
    try:
        start = perf_counter()
        bytes_read = 0
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(64 * 1024):
                    bytes_read += len(chunk)
                    if bytes_read >= sample_bytes:
                        break

        elapsed = max(perf_counter() - start, 1e-6)
        throughput = bytes_read / elapsed
        result.update(
            {
                "bytes_read": bytes_read,
                "seconds": elapsed,
                "bytes_per_second": throughput,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    log_event("tool.estimate_download_time", result)
    return result


@function_tool
async def check_zenodo_record(identifier: str) -> dict:
    return await check_zenodo_record_request(identifier)


@function_tool
async def check_figshare_record(identifier: str) -> dict:
    return await check_figshare_record_request(identifier)


@function_tool
async def estimate_download_time(url: str, sample_bytes: int = 1024 * 1024) -> dict:
    return await estimate_download_time_request(url, sample_bytes)
