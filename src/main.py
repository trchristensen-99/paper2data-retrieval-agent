from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.agents.manager import run_pipeline
from src.utils.env import load_env_file
from src.utils.network import check_external_service_access, check_openai_dns


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Paper2Data retrieval agent pipeline")
    parser.add_argument("paper_markdown", type=Path, help="Path to paper markdown file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated outputs",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip expensive deep checks (data availability, QC, enrichment/repair retries)",
    )
    parser.add_argument(
        "--strict-network",
        action="store_true",
        help="Require external service preflight checks to pass before running.",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, str]:
    if not args.paper_markdown.exists():
        raise FileNotFoundError(f"Input file not found: {args.paper_markdown}")

    paper_text = args.paper_markdown.read_text(encoding="utf-8")
    artifacts = await run_pipeline(paper_text, fast_mode=bool(args.fast))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    structured_path = run_dir / "structured_record.json"
    report_path = run_dir / "retrieval_report.md"
    log_path = run_dir / "retrieval_log.md"
    timings_path = run_dir / "step_timings.json"

    structured_path.write_text(
        json.dumps(artifacts.record.model_dump(), indent=2), encoding="utf-8"
    )
    report_path.write_text(artifacts.retrieval_report_markdown, encoding="utf-8")
    log_path.write_text(artifacts.retrieval_log_markdown, encoding="utf-8")
    timings_path.write_text(
        json.dumps(
            {
                "step_timings_seconds": artifacts.step_timings_seconds,
                "pipeline_duration_seconds": artifacts.pipeline_duration_seconds,
                "quality_notes": artifacts.quality_notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "structured_record": str(structured_path),
        "retrieval_report": str(report_path),
        "retrieval_log": str(log_path),
        "step_timings": str(timings_path),
        "confidence": f"{artifacts.record.extraction_confidence:.2f}",
        "title": artifacts.record.metadata.title,
        "pipeline_duration_seconds": f"{artifacts.pipeline_duration_seconds:.2f}",
    }


def main() -> None:
    load_env_file()
    ok, msg = check_openai_dns()
    if not ok:
        raise RuntimeError(
            f"{msg}. Fix DNS/network and retry. "
            "Try: nslookup api.openai.com, then set DNS to 1.1.1.1 or 8.8.8.8."
        )
    parser = _build_arg_parser()
    args = parser.parse_args()
    strict_network = bool(args.strict_network)
    svc_ok, svc_msg, checks = check_external_service_access()
    if strict_network and not svc_ok:
        failure_lines = [
            f"{c['name']}: status={c['status_code']} error={c['error']}"
            for c in checks
            if not c["ok"]
        ]
        raise RuntimeError(
            f"{svc_msg}\n" + "\n".join(failure_lines) + "\n"
            "Set proxy/DNS and retry, or run without --strict-network."
        )
    if not svc_ok:
        print(f"[network] WARNING: {svc_msg}. Continuing without strict enforcement.", flush=True)
    summary = asyncio.run(_run(args))

    print("Paper2Data retrieval pipeline complete")
    print(f"Title: {summary['title']}")
    print(f"Confidence: {summary['confidence']}")
    print(f"Structured record: {summary['structured_record']}")
    print(f"Report: {summary['retrieval_report']}")
    print(f"Retrieval log: {summary['retrieval_log']}")
    print(f"Step timings: {summary['step_timings']}")
    print(f"Pipeline duration (s): {summary['pipeline_duration_seconds']}")


if __name__ == "__main__":
    main()
