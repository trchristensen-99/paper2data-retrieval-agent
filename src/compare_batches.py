from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = payload.get("papers", [])
    durations = [p.get("duration_seconds") for p in papers if p.get("duration_seconds")]
    payload["derived"] = {
        "avg_duration_seconds": round(mean(durations), 2) if durations else None,
        "max_duration_seconds": max(durations) if durations else None,
        "min_duration_seconds": min(durations) if durations else None,
    }
    return payload


def _find_latest_summary(root: Path) -> Path | None:
    candidates = sorted(root.glob("**/batch_summary.json"))
    return candidates[-1] if candidates else None


def _compare(a: dict, b: dict) -> dict:
    am = a.get("metrics", {})
    bm = b.get("metrics", {})
    ad = a.get("derived", {})
    bd = b.get("derived", {})
    return {
        "papers_total_delta": bm.get("papers_total", 0) - am.get("papers_total", 0),
        "papers_success_delta": bm.get("papers_success", 0) - am.get("papers_success", 0),
        "papers_failed_delta": bm.get("papers_failed", 0) - am.get("papers_failed", 0),
        "critical_coverage_delta": round(
            bm.get("critical_coverage_percent", 0.0) - am.get("critical_coverage_percent", 0.0), 2
        ),
        "doi_pmid_coverage_delta": round(
            bm.get("doi_pmid_coverage_percent", 0.0) - am.get("doi_pmid_coverage_percent", 0.0), 2
        ),
        "avg_duration_seconds_delta": (
            round((bd.get("avg_duration_seconds") or 0) - (ad.get("avg_duration_seconds") or 0), 2)
            if bd.get("avg_duration_seconds") is not None and ad.get("avg_duration_seconds") is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Paper2Data batch summaries")
    parser.add_argument("--baseline", type=Path, help="Path to baseline batch_summary.json")
    parser.add_argument("--updated", type=Path, help="Path to updated batch_summary.json")
    parser.add_argument(
        "--comparisons-root",
        type=Path,
        default=Path("outputs/comparisons"),
        help="Root used to auto-discover latest summary if args omitted",
    )
    args = parser.parse_args()

    baseline_path = args.baseline
    updated_path = args.updated

    if baseline_path is None:
        baseline_path = _find_latest_summary(args.comparisons_root / "baseline")
    if updated_path is None:
        updated_path = _find_latest_summary(args.comparisons_root / "updated_full")

    if baseline_path is None or not baseline_path.exists():
        raise SystemExit("Baseline summary not found. Pass --baseline explicitly.")
    if updated_path is None or not updated_path.exists():
        raise SystemExit("Updated summary not found. Pass --updated explicitly.")

    baseline = _load_summary(baseline_path)
    updated = _load_summary(updated_path)

    output = {
        "baseline_summary": str(baseline_path),
        "updated_summary": str(updated_path),
        "baseline_metrics": baseline.get("metrics", {}),
        "updated_metrics": updated.get("metrics", {}),
        "baseline_derived": baseline.get("derived", {}),
        "updated_derived": updated.get("derived", {}),
        "delta": _compare(baseline, updated),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
