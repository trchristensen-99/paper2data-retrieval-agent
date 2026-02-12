from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.schemas.models import (
    DataAvailabilityReport,
    MetadataRecord,
    MethodsSummary,
    QualityCheckOutput,
    ResultsSummary,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    metadata: float
    methods: float
    results: float
    data_access: float
    penalties: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metadata": self.metadata,
            "methods": self.methods,
            "results": self.results,
            "data_access": self.data_access,
            "penalties": self.penalties,
        }


def compute_extraction_confidence(
    *,
    metadata: MetadataRecord,
    methods: MethodsSummary,
    results: ResultsSummary,
    data_availability: DataAvailabilityReport,
    quality_check: QualityCheckOutput,
) -> ConfidenceBreakdown:
    metadata_component = 0.0
    if metadata.title.strip():
        metadata_component += 0.08
    if metadata.authors:
        metadata_component += 0.08
    if metadata.doi or metadata.pmid:
        metadata_component += 0.08
    if metadata.journal:
        metadata_component += 0.04

    methods_component = 0.0
    if methods.experimental_design.strip():
        methods_component += 0.08
    if methods.assay_types:
        methods_component += 0.04
    if methods.sample_sizes:
        methods_component += 0.04
    if methods.statistical_tests:
        methods_component += 0.04

    results_component = 0.0
    n_quant = len(results.quantitative_findings)
    if n_quant >= 3:
        results_component += 0.15
    elif n_quant > 0:
        results_component += 0.10
    if results.spin_assessment.strip():
        results_component += 0.05
    if results.qualitative_findings:
        results_component += 0.04

    data_access_component = 0.0
    status = (data_availability.overall_status or "").strip().lower()
    status_scores = {
        "accessible": 0.10,
        "partially_accessible": 0.07,
        "unavailable": 0.04,
        "not_checked": 0.0,
    }
    data_access_component += status_scores.get(status, 0.03)
    if data_availability.verified_repositories:
        data_access_component += 0.05

    penalties = 0.0
    penalties += min(len(quality_check.missing_fields), 6) * 0.025
    penalties += min(len(quality_check.suspicious_empty_fields), 6) * 0.015

    raw = (
        metadata_component
        + methods_component
        + results_component
        + data_access_component
        + 0.12  # base score for successful end-to-end extraction
        - penalties
    )
    score = _clamp(raw, lo=0.05, hi=0.99)
    return ConfidenceBreakdown(
        score=score,
        metadata=metadata_component,
        methods=methods_component,
        results=results_component,
        data_access=data_access_component,
        penalties=penalties,
    )
