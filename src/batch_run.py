from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.agents.manager import PipelineArtifacts, run_pipeline
from src.utils.env import load_env_file
from src.utils.network import check_openai_dns

CRITICAL_FIELDS = [
    "metadata.title",
    "metadata.authors",
    "metadata.journal",
    "methods.assay_types",
    "methods.sample_sizes",
    "methods.experimental_design",
    "results.quantitative_findings",
    "results.spin_assessment",
    "data_availability.overall_status",
    "data_availability.notes",
]
OPTIONAL_TRACKED_FIELDS = ["metadata.doi", "metadata.pmid"]


@dataclass
class PaperRunSummary:
    paper_id: str
    input_path: str
    status: str
    output_dir: str | None
    duration_seconds: float | None
    missing_critical_fields: list[str]
    missing_optional_tracked_fields: list[str]
    error: str | None


def _get_field(payload: dict, path: str):
    cur = payload
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _find_markdown_inputs(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/data_for_retrieval_agent/*.md"))


async def _run_one(input_path: Path, run_root: Path) -> PaperRunSummary:
    paper_id = input_path.stem
    paper_out = run_root / paper_id
    paper_out.mkdir(parents=True, exist_ok=True)

    try:
        paper_text = input_path.read_text(encoding="utf-8")

        last_error: Exception | None = None
        artifacts: PipelineArtifacts | None = None
        for attempt in range(1, 4):
            try:
                artifacts = await run_pipeline(paper_text)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc).lower()
                transient = any(
                    token in msg
                    for token in (
                        "connection error",
                        "temporarily unavailable",
                        "timed out",
                        "nodename nor servname provided",
                    )
                )
                if not transient or attempt == 3:
                    raise
                wait_seconds = 5 * attempt
                print(
                    f"[batch] transient_error paper_id={paper_id} attempt={attempt} "
                    f"wait_seconds={wait_seconds} error={exc}",
                    flush=True,
                )
                await asyncio.sleep(wait_seconds)
        if artifacts is None and last_error is not None:
            raise last_error

        structured_path = paper_out / "structured_record.json"
        report_path = paper_out / "retrieval_report.md"
        log_path = paper_out / "retrieval_log.md"
        timings_path = paper_out / "step_timings.json"

        structured_payload = artifacts.record.model_dump()
        structured_path.write_text(json.dumps(structured_payload, indent=2), encoding="utf-8")
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

        missing_critical = [f for f in CRITICAL_FIELDS if _is_missing(_get_field(structured_payload, f))]
        missing_optional = [f for f in OPTIONAL_TRACKED_FIELDS if _is_missing(_get_field(structured_payload, f))]

        return PaperRunSummary(
            paper_id=paper_id,
            input_path=str(input_path),
            status="ok",
            output_dir=str(paper_out),
            duration_seconds=artifacts.pipeline_duration_seconds,
            missing_critical_fields=missing_critical,
            missing_optional_tracked_fields=missing_optional,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return PaperRunSummary(
            paper_id=paper_id,
            input_path=str(input_path),
            status="error",
            output_dir=None,
            duration_seconds=None,
            missing_critical_fields=[],
            missing_optional_tracked_fields=[],
            error=str(exc),
        )


async def _run_batch(input_root: Path, output_root: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"batch_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    inputs = _find_markdown_inputs(input_root)
    summaries: list[PaperRunSummary] = []

    for idx, path in enumerate(inputs, start=1):
        print(f"[batch] {idx}/{len(inputs)} {path}", flush=True)
        summary = await _run_one(path, run_root)
        summaries.append(summary)
        print(
            f"[batch] done paper_id={summary.paper_id} status={summary.status} "
            f"duration={summary.duration_seconds}",
            flush=True,
        )

    ok_runs = [s for s in summaries if s.status == "ok"]
    critical_total = len(CRITICAL_FIELDS) * len(ok_runs)
    critical_missing = sum(len(s.missing_critical_fields) for s in ok_runs)
    optional_total = len(OPTIONAL_TRACKED_FIELDS) * len(ok_runs)
    optional_missing = sum(len(s.missing_optional_tracked_fields) for s in ok_runs)

    metrics = {
        "papers_total": len(summaries),
        "papers_success": len(ok_runs),
        "papers_failed": len(summaries) - len(ok_runs),
        "critical_coverage_percent": round(
            100.0 * (critical_total - critical_missing) / critical_total, 2
        )
        if critical_total
        else 0.0,
        "doi_pmid_coverage_percent": round(
            100.0 * (optional_total - optional_missing) / optional_total, 2
        )
        if optional_total
        else 0.0,
    }

    payload = {
        "run_root": str(run_root),
        "metrics": metrics,
        "papers": [asdict(s) for s in summaries],
    }

    summary_path = run_root / "batch_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[batch] summary={summary_path}", flush=True)
    print(f"[batch] metrics={metrics}", flush=True)
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval pipeline across a folder of paper markdown files")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("../data_for_agents_example/data30_final"),
        help="Path containing per-paper subfolders with data_for_retrieval_agent/*.md",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory where batch run artifacts are written",
    )
    return parser


def main() -> None:
    load_env_file()
    ok, msg = check_openai_dns()
    if not ok:
        raise RuntimeError(
            f"{msg}. Fix DNS/network and retry batch. "
            "Try: nslookup api.openai.com, then set DNS to 1.1.1.1 or 8.8.8.8."
        )
    args = _build_parser().parse_args()
    summary_path = asyncio.run(_run_batch(args.input_root, args.output_root))
    print(f"Batch run complete: {summary_path}")


if __name__ == "__main__":
    main()
