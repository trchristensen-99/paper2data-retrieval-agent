from __future__ import annotations

import pytest

from src.agents import manager
from src.schemas.models import (
    DataAccession,
    DataAvailabilityReport,
    DescriptiveStat,
    Finding,
    FigureSummary,
    MetadataRecord,
    MethodBenchmark,
    MethodsSummary,
    PaperAnatomyOutput,
    PaperRecord,
    ResultsSummary,
    SynthesisOutput,
)


def test_schema_defaults() -> None:
    metadata = MetadataRecord(title="A", authors=["X"])
    assert metadata.keywords == []


def test_methods_summary_shape() -> None:
    methods = MethodsSummary(
        organisms=["human"],
        cell_types=["lymphocyte"],
        assay_types=["RNA-seq"],
        sample_sizes={"group_a": "n=10"},
        statistical_tests=["Wilcoxon rank-sum"],
        experimental_design="Case-control",
        methods_completeness="adequate",
    )
    assert methods.sample_sizes["group_a"] == "n=10"


def test_results_summary_shape() -> None:
    summary = ResultsSummary(
        paper_type="review",
        synthesized_claims=["Signal present in treated cohort"],
    )
    assert summary.synthesized_claims


def test_data_availability_values() -> None:
    report = DataAvailabilityReport(
        overall_status="not_checked",
        claimed_repositories=["GEO"],
        verified_repositories=[],
        discrepancies=[],
        notes="No checks run",
    )
    assert report.overall_status == "not_checked"


@pytest.mark.asyncio
async def test_run_pipeline_with_mocked_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _metadata(_: str, guidance: str | None = None) -> MetadataRecord:
        return MetadataRecord(title="Demo Paper", authors=["A. Author"])

    async def _anatomy(_: str) -> PaperAnatomyOutput:
        return PaperAnatomyOutput(
            sections=["Abstract", "Methods"],
            tables=["Table 1"],
            figures=["Figure 1"],
            urls=["https://example.org"],
            accession_candidates=["GSE00001"],
            prisma_flow={"included": 1},
            notes="ok",
        )

    async def _methods(_: str, guidance: str | None = None) -> MethodsSummary:
        return MethodsSummary(
            organisms=["human"],
            cell_types=["HEK293"],
            assay_types=["RNA-seq"],
            sample_sizes={"control": "n=3", "treated": "n=3"},
            statistical_tests=["DESeq2 Wald test"],
            experimental_design="Two-group differential expression design.",
            methods_completeness="sufficient",
        )

    async def _results(_: str, paper_type: str | None = None, guidance: str | None = None) -> ResultsSummary:
        return ResultsSummary(
            paper_type=paper_type or "experimental",
            experimental_findings=[
                Finding(
                    claim="Gene X increased in treated samples",
                    metric="log2 fold change",
                    value="1.4",
                    effect_size="1.4",
                    context="Treated vs control",
                    confidence=0.8,
                )
            ],
            dataset_properties=[DescriptiveStat(property="n_records", value="6", context="demo")],
            synthesized_claims=["Primary pattern is treatment-associated increase."],
            method_benchmarks=[MethodBenchmark(task="demo", metric="acc", value="0.9", context="test")],
            key_figures=[FigureSummary(figure_id="Fig1", description="Volcano plot", key_findings=["Gene X up"])],
        )

    async def _data(_: str, paper_type: str | None = None, guidance: str | None = None):
        class _Out:
            data_accessions = [
                DataAccession(
                    accession_id="GSE00001",
                    category="supplementary_data",
                    repository="GEO",
                    description="RNA-seq cohort",
                    is_accessible=True,
                    file_count=2,
                    files_listed=["a.fastq.gz", "b.fastq.gz"],
                )
            ]
            related_resources = []
            data_availability = DataAvailabilityReport(
                overall_status="accessible",
                claimed_repositories=["GEO"],
                verified_repositories=["GEO"],
                discrepancies=[],
                notes="All checked resources responded.",
            )

        return _Out()

    async def _synthesis(payload):
        record = PaperRecord(
            metadata=payload.metadata,
            methods=payload.methods,
            results=payload.results,
            data_accessions=payload.data_accessions,
            data_availability=payload.data_availability,
            extraction_timestamp="2026-02-12T00:00:00",
            extraction_confidence=0.85,
        )
        return SynthesisOutput(
            record=record,
            retrieval_report_markdown="# Retrieval Report",
            retrieval_log_markdown="# Retrieval Log",
        )

    class _Enrich:
        doi = None
        pmid = None
        journal = None
        publication_date = None
        notes = "No enrichment needed"

    async def _enrichment(*args, **kwargs):
        return _Enrich()

    class _QC:
        should_retry = False
        retry_instructions = []
        missing_fields = []
        suspicious_empty_fields = []
        notes = "No issues detected"

    async def _quality(*args, **kwargs):
        return _QC()

    monkeypatch.setattr(manager, "run_metadata_agent", _metadata)
    monkeypatch.setattr(manager, "run_anatomy_agent", _anatomy)
    monkeypatch.setattr(manager, "run_methods_agent", _methods)
    monkeypatch.setattr(manager, "run_results_agent", _results)
    monkeypatch.setattr(manager, "run_data_availability_agent", _data)
    monkeypatch.setattr(manager, "run_quality_control_agent", _quality)
    monkeypatch.setattr(manager, "run_metadata_enrichment_agent", _enrichment)
    monkeypatch.setattr(manager, "run_synthesis_agent", _synthesis)

    artifacts = await manager.run_pipeline("paper body")
    assert artifacts.record.metadata.title == "Demo Paper"
    assert 0.0 <= artifacts.record.extraction_confidence <= 1.0
