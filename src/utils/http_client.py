from __future__ import annotations

import os

import httpx


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _effective_timeout(default_seconds: float) -> float:
    raw = (os.getenv("P2D_HTTP_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return default_seconds
    try:
        value = float(raw)
    except Exception:  # noqa: BLE001
        return default_seconds
    return value if value > 0 else default_seconds


def _client_kwargs(timeout_seconds: float, follow_redirects: bool) -> dict:
    kwargs: dict = {
        "timeout": _effective_timeout(timeout_seconds),
        "follow_redirects": follow_redirects,
        "trust_env": True,
        "verify": _env_bool("P2D_SSL_VERIFY", True),
    }
    proxy = (os.getenv("P2D_HTTP_PROXY") or "").strip()
    if proxy:
        kwargs["proxy"] = proxy
    user_agent = (os.getenv("P2D_USER_AGENT") or "").strip()
    if user_agent:
        kwargs["headers"] = {"User-Agent": user_agent}
    return kwargs


def make_async_client(timeout_seconds: float, follow_redirects: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(**_client_kwargs(timeout_seconds, follow_redirects))


def make_sync_client(timeout_seconds: float, follow_redirects: bool = True) -> httpx.Client:
    return httpx.Client(**_client_kwargs(timeout_seconds, follow_redirects))

