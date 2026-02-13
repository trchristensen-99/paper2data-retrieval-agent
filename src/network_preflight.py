from __future__ import annotations

import argparse
import json
import sys

from src.utils.env import load_env_file
from src.utils.network import check_external_service_access, check_openai_dns


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preflight check for external services used by Paper2Data tools")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any service check fails")
    p.add_argument("--timeout", type=float, default=12.0, help="HTTP timeout seconds for each probe")
    return p


def main() -> None:
    load_env_file()
    args = _parser().parse_args()
    dns_ok, dns_msg = check_openai_dns()
    svc_ok, svc_msg, checks = check_external_service_access(timeout_seconds=float(args.timeout))
    ok = dns_ok and svc_ok

    payload = {
        "ok": ok,
        "dns_ok": dns_ok,
        "dns_message": dns_msg,
        "services_ok": svc_ok,
        "services_message": svc_msg,
        "checks": checks,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(dns_msg)
        print(svc_msg)
        for c in checks:
            status = c["status_code"] if c["status_code"] is not None else "n/a"
            err = c["error"] or "-"
            print(f"- {c['name']}: ok={c['ok']} status={status} error={err}")

    if args.strict and not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()

