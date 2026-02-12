from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.database.reviewer import parse_sections, review_and_update_entry
from src.database.store import PaperDatabase
from src.utils.env import load_env_file


def _find_structured_records(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("**/structured_record.json"))


async def _cmd_ingest(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        paths = _find_structured_records(args.input)
        if not paths:
            print(f"No structured_record.json files found under {args.input}")
            return
        results = await db.ingest_many(paths)
        inserted = sum(1 for r in results if r.action == "inserted")
        updated = sum(1 for r in results if r.action == "updated")
        print(f"Ingested {len(results)} records | inserted={inserted} updated={updated}")
        for r in results:
            print(f"- {r.action.upper()} paper_id={r.paper_id} merged={r.merged}")
    finally:
        db.close()


def _cmd_query(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        rows = db.search(args.q, limit=args.limit)
        print(json.dumps(rows, indent=2))
    finally:
        db.close()


def _cmd_show(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        row = db.fetch_paper(args.paper_id)
        if row is None:
            print(f"paper_id not found: {args.paper_id}")
            return
        print(json.dumps(row, indent=2))
    finally:
        db.close()


def _cmd_stats(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        print(json.dumps(db.stats(), indent=2))
    finally:
        db.close()


def _cmd_init(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        print(json.dumps(db.stats(), indent=2))
    finally:
        db.close()


async def _cmd_review_update(args: argparse.Namespace) -> None:
    db = PaperDatabase(args.db)
    try:
        sections = parse_sections(args.sections)
        result = await review_and_update_entry(
            db=db,
            paper_id=args.paper_id,
            paper_markdown_path=args.paper_markdown,
            sections=sections,
        )
        print(
            json.dumps(
                {
                    "paper_id": result.paper_id,
                    "sections_updated": result.sections_updated,
                    "changed_top_level_fields": result.changed_top_level_fields,
                    "confidence_before": result.confidence_before,
                    "confidence_after": result.confidence_after,
                },
                indent=2,
            )
        )
    finally:
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper2Data queryable database CLI")
    p.add_argument("--db", type=str, default="outputs/paper_terminal.db", help="SQLite DB path")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Create database if missing")
    init_p.set_defaults(func=lambda a: _cmd_init(a))

    ingest_p = sub.add_parser("ingest", help="Ingest structured_record.json files")
    ingest_p.add_argument("--input", type=Path, required=True, help="File or directory to ingest")
    ingest_p.set_defaults(func=lambda a: _cmd_ingest(a))

    query_p = sub.add_parser("query", help="Search papers")
    query_p.add_argument("--q", type=str, required=True, help="Free-text search query")
    query_p.add_argument("--limit", type=int, default=20)
    query_p.set_defaults(func=lambda a: _cmd_query(a))

    show_p = sub.add_parser("show", help="Show one paper by id")
    show_p.add_argument("--paper-id", type=str, required=True)
    show_p.set_defaults(func=lambda a: _cmd_show(a))

    stats_p = sub.add_parser("stats", help="Database summary")
    stats_p.set_defaults(func=lambda a: _cmd_stats(a))

    review_p = sub.add_parser(
        "review-update",
        help="Re-run extraction on source paper and selectively update an existing DB entry",
    )
    review_p.add_argument("--paper-id", type=str, required=True)
    review_p.add_argument("--paper-markdown", type=Path, required=True)
    review_p.add_argument(
        "--sections",
        type=str,
        default="all",
        help="Comma-separated: metadata,methods,results,data_accessions,data_availability,code_repositories (or 'all')",
    )
    review_p.set_defaults(func=lambda a: _cmd_review_update(a))

    return p


def main() -> None:
    load_env_file()
    args = _build_parser().parse_args()
    result = args.func(args)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
