from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.agents.manager import run_pipeline
from src.utils.env import load_env_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Paper2Data retrieval agent pipeline")
    parser.add_argument("paper_markdown", type=Path, help="Path to paper markdown file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated outputs",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, str]:
    if not args.paper_markdown.exists():
        raise FileNotFoundError(f"Input file not found: {args.paper_markdown}")

    paper_text = args.paper_markdown.read_text(encoding="utf-8")
    artifacts = await run_pipeline(paper_text)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    structured_path = run_dir / "structured_record.json"
    report_path = run_dir / "retrieval_report.md"
    log_path = run_dir / "retrieval_log.md"

    structured_path.write_text(
        json.dumps(artifacts.record.model_dump(), indent=2), encoding="utf-8"
    )
    report_path.write_text(artifacts.retrieval_report_markdown, encoding="utf-8")
    log_path.write_text(artifacts.retrieval_log_markdown, encoding="utf-8")

    return {
        "structured_record": str(structured_path),
        "retrieval_report": str(report_path),
        "retrieval_log": str(log_path),
        "confidence": f"{artifacts.record.extraction_confidence:.2f}",
        "title": artifacts.record.metadata.title,
    }


def main() -> None:
    load_env_file()
    parser = _build_arg_parser()
    args = parser.parse_args()
    summary = asyncio.run(_run(args))

    print("Paper2Data retrieval pipeline complete")
    print(f"Title: {summary['title']}")
    print(f"Confidence: {summary['confidence']}")
    print(f"Structured record: {summary['structured_record']}")
    print(f"Report: {summary['retrieval_report']}")
    print(f"Retrieval log: {summary['retrieval_log']}")


if __name__ == "__main__":
    main()
