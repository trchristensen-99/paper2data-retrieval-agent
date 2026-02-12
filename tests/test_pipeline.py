from __future__ import annotations

from src.schemas.models import (
    DataAvailabilityReport,
    MetadataRecord,
    MethodsSummary,
    ResultsSummary,
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
        quantitative_findings=[],
        qualitative_findings=["Signal present in treated cohort"],
        spin_assessment="mostly aligned",
    )
    assert summary.spin_assessment


def test_data_availability_values() -> None:
    report = DataAvailabilityReport(
        overall_status="not_checked",
        claimed_repositories=["GEO"],
        verified_repositories=[],
        discrepancies=[],
        notes="No checks run",
    )
    assert report.overall_status == "not_checked"
