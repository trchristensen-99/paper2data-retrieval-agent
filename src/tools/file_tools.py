from __future__ import annotations

import asyncio
from ftplib import FTP
from urllib.parse import urlparse

from agents import function_tool

from src.utils.logging import log_event


def _list_ftp_files_sync(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"Invalid FTP URL: {url}")

    path = parsed.path or "/"
    files: list[str] = []
    with FTP(host, timeout=15) as ftp:
        ftp.login()
        ftp.cwd(path)
        ftp.retrlines("NLST", files.append)
    return files


@function_tool
async def list_ftp_files(url: str) -> list[str]:
    """List files available at an FTP URL."""
    try:
        files = await asyncio.to_thread(_list_ftp_files_sync, url)
        log_event("tool.list_ftp_files", {"url": url, "file_count": len(files)})
        return files
    except Exception as exc:  # noqa: BLE001
        log_event("tool.list_ftp_files", {"url": url, "error": str(exc)})
        return []
