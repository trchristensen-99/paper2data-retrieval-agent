from __future__ import annotations

import pytest

from src.agents import manager
from src.agents.data_availability import _enrich_accession
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


@pytest.mark.asyncio
async def test_dataset_descriptor_backfills_profile_and_code(monkeypatch: pytest.MonkeyPatch) -> None:
    paper = """
    # Responsible AI Measures Dataset
    Published online: 20 December 2025
    The dataset contains 791 measures and 12,067 data points in 16 columns.
    Publication years range from 2011-2023 with 257 papers included.
    Data are available at https://www.fgshare.com/articles/dataset/example/29551001
    Repository includes README.md, dataset.xlsx, processing_notebook.ipynb, visualize.py, Sunburst_Visualization_Link.md
    """

    async def _metadata(_: str, guidance: str | None = None) -> MetadataRecord:
        return MetadataRecord(
            title="Responsible AI Measures Dataset",
            authors=["A. Author"],
            paper_type="dataset_descriptor",
            publication_date="2025",
            journal="Nature Scientific Data",
        )

    async def _anatomy(_: str) -> PaperAnatomyOutput:
        return PaperAnatomyOutput(notes="ok")

    async def _methods(_: str, guidance: str | None = None) -> MethodsSummary:
        return MethodsSummary(
            organisms=[],
            cell_types=[],
            assay_types=["scoping_review"],
            sample_sizes={},
            statistical_tests=[],
            experimental_design="scoping review",
            methods_completeness="sufficient",
        )

    async def _results(_: str, paper_type: str | None = None, guidance: str | None = None) -> ResultsSummary:
        return ResultsSummary(paper_type=paper_type, dataset_properties=[], dataset_profile=None)

    async def _data(_: str, paper_type: str | None = None, guidance: str | None = None):
        class _Out:
            data_accessions = [
                DataAccession(
                    accession_id="10.6084/m9.fgshare.29551001",
                    category="primary_dataset",
                    repository="Figshare",
                    url="https://www.fgshare.com/articles/dataset/example/29551001",
                    description="dataset",
                    is_accessible=True,
                )
            ]
            related_resources = []
            data_availability = DataAvailabilityReport(
                overall_status="accessible",
                claimed_repositories=["Figshare"],
                verified_repositories=["Figshare"],
                discrepancies=[],
                notes="ok",
                check_status="ok",
            )

        return _Out()

    async def _synthesis(payload):
        record = PaperRecord(
            metadata=payload.metadata,
            methods=payload.methods,
            results=payload.results,
            data_accessions=payload.data_accessions,
            data_availability=payload.data_availability,
            code_repositories=payload.code_repositories,
            extraction_timestamp="2026-02-12T00:00:00",
            extraction_confidence=0.5,
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

    artifacts = await manager.run_pipeline(paper)
    assert artifacts.record.results.dataset_profile is not None
    assert artifacts.record.results.dataset_profile.record_count == 791
    assert artifacts.record.metadata.publication_date == "2025-12-20"
    assert any("figshare.com" in (a.url or "") for a in artifacts.record.data_accessions)
    assert artifacts.record.code_available is True
    assert artifacts.record.archival_repositories
    assert artifacts.record.methods.experimental_design_steps
    assert artifacts.record.methods.assay_type_mappings
    assert any((m.ontology_id or "") for m in artifacts.record.methods.assay_type_mappings)
    assert any((a.normalized_id or "").startswith("doi:") for a in artifacts.record.data_accessions)
    assert any((a.system or "") == "DOI" for a in artifacts.record.data_accessions)
    assert artifacts.record.results.experimental_findings


def test_table_block_extraction_includes_rows() -> None:
    text = """
    <table>
      <tr><th>Variable</th><th>Description</th></tr>
      <tr><td>Measure</td><td>The name of the measure</td></tr>
      <tr><td>Principle</td><td>Ethical principle</td></tr>
    </table>
    """
    tables = manager._extract_table_blocks(text)
    assert tables
    assert tables[0].columns == ["Variable", "Description"]
    assert tables[0].data
    assert tables[0].data[0]["Variable"] == "Measure"
    assert tables[0].provenance is not None


def test_table_topology_normalizer_adds_category_and_dedupes() -> None:
    raw = [
        manager.ExtractedTable(
            table_id="Table 2",
            columns=["journal", "count", "percent"],
            data=[
                {"journal": "New Disease Description", "count": "", "percent": ""},
                {"journal": "Am. J. Hum. Genet.", "count": "457", "percent": "14.9"},
                {"journal": "New Gene Discovery", "count": "", "percent": ""},
                {"journal": "Nature Genetics", "count": "1137", "percent": "26.7"},
            ],
            key_content=[],
        ),
        manager.ExtractedTable(
            table_id="Table 2-partA",
            columns=["category", "journal", "count", "percent"],
            data=[{"category": "New Disease Description", "journal": "Am. J. Hum. Genet.", "count": "457", "percent": "14.9"}],
            key_content=[],
        ),
    ]
    out = manager._normalize_table_topology(raw)
    ids = {t.table_id for t in out}
    assert "Table 2-partA" in ids
    assert "Table 2" not in ids


@pytest.mark.asyncio
async def test_enrich_accession_repairs_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _check_url(url: str):
        if "fle=" in url:
            return {"is_accessible": False, "status_code": 404}
        if "file=" in url:
            return {"is_accessible": True, "status_code": 200}
        return {"is_accessible": False, "status_code": 404}

    monkeypatch.setattr("src.agents.data_availability.check_url_request", _check_url)
    accession = DataAccession(
        accession_id="A1",
        repository="Figshare",
        category="primary_dataset",
        url="https://figshare.com/articles/dataset/example/29551001?fle=57701437",
        description="test",
    )
    out = await _enrich_accession(accession)
    assert out.is_accessible is True
    assert out.url_repaired is True
    assert "file=57701437" in (out.url or "")


def test_data_asset_derivation_from_accessions() -> None:
    accessions = [
        DataAccession(
            accession_id="10.6084/m9.figshare.29551001",
            category="primary_dataset",
            repository="Figshare",
            url="https://figshare.com/articles/dataset/example/29551001",
            description="dataset",
            files_listed=["README.md", "Gene-RD-Provenance_V2.1.txt"],
            is_accessible=True,
        )
    ]
    profile = manager.DatasetProfile(record_count=4565)
    assets = manager._derive_data_assets(
        data_accessions=accessions,
        dataset_profile=profile,
        paper_markdown="Gene-RD-Provenance_V2.1.txt is provided.",
    )
    assert len(assets) >= 2
    assert any(a.content_type == "gene_disease_associations" for a in assets)


def test_partition_results_findings_vs_properties_dedupes() -> None:
    results = ResultsSummary(
        paper_type="dataset_descriptor",
        dataset_properties=[
            DescriptiveStat(property="record_count", value="4565", unit="rows", context="dataset size"),
            DescriptiveStat(property="disease_mesh_mapping", value="58.7%", unit="percent", context="coverage"),
        ],
        experimental_findings=[
            Finding(
                claim="record_count = 4565",
                metric="record_count",
                value="4565",
                unit="rows",
                context="dataset size",
                confidence=0.8,
            )
        ],
    )
    manager._partition_results_findings_vs_properties(results)
    assert all((f.metric or "") != "record_count" for f in results.experimental_findings)
    assert any("58.7%" in (f.value or "") for f in results.experimental_findings)


def test_locate_provenance_resolves_source_page() -> None:
    text = "Page 1\nIntro line\n\fPage 2\nTarget value appears here"
    prov = manager._locate_provenance(text, "Target value")
    assert prov is not None
    assert prov.source_page == 2
