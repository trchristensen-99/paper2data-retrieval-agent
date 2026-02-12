from __future__ import annotations

import socket


def check_openai_dns(host: str = "api.openai.com") -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return True, f"DNS resolution succeeded for {host}"
    except Exception as exc:  # noqa: BLE001
        return False, f"DNS resolution failed for {host}: {exc}"
