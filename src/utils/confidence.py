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
    # Explicit rubric (deterministic, not model "vibes"):
    # 1) Metadata completeness/identifiability (30%)
    # 2) Methods reproducibility detail (25%)
    # 3) Results quantitative fidelity (25%)
    # 4) Data availability verification (10%)
    # 5) QC cleanliness (10%)

    metadata_checks = [
        bool(metadata.title.strip()),
        bool(metadata.authors),
        bool((metadata.journal or "").strip()) and not (metadata.journal or "").strip().lower().startswith("unknown"),
        bool((metadata.publication_date or "").strip()) and not (metadata.publication_date or "").strip().lower().startswith("unknown"),
        bool(metadata.keywords),
        bool(metadata.doi or metadata.pmid),
    ]
    metadata_quality = sum(1.0 for ok in metadata_checks if ok) / len(metadata_checks)
    metadata_component = metadata_quality * 0.30

    methods_checks = [
        bool(methods.experimental_design.strip()),
        bool(methods.assay_types),
        bool(methods.sample_sizes),
        bool(methods.statistical_tests),
        bool(methods.organisms or methods.cell_types),
    ]
    methods_quality = sum(1.0 for ok in methods_checks if ok) / len(methods_checks)
    methods_component = methods_quality * 0.25

    quantitative = results.quantitative_findings
    if quantitative:
        finding_scores: list[float] = []
        for f in quantitative:
            checks = [
                bool(f.claim.strip()),
                bool(f.metric.strip()),
                bool(f.value.strip()),
                bool(f.context.strip()),
                bool((f.effect_size or "").strip() or (f.confidence_interval or "").strip()),
            ]
            finding_scores.append(sum(1.0 for ok in checks if ok) / len(checks))
        quant_quality = sum(finding_scores) / len(finding_scores)
    else:
        quant_quality = 0.0

    spin_text = (results.spin_assessment or "").strip().lower()
    spin_quality = 1.0 if spin_text and spin_text != "not_assessed" else 0.0
    qual_quality = 1.0 if results.qualitative_findings else 0.0
    results_quality = _clamp((0.70 * quant_quality) + (0.20 * spin_quality) + (0.10 * qual_quality))
    results_component = results_quality * 0.25

    status = (data_availability.overall_status or "").strip().lower()
    status_scores = {
        "accessible": 1.0,
        "partially_accessible": 0.7,
        "unavailable": 0.3,
        "not_checked": 0.0,
    }
    status_quality = status_scores.get(status, 0.0)
    verified_bonus = 0.2 if data_availability.verified_repositories else 0.0
    discrepancy_penalty = min(0.3, 0.05 * len(data_availability.discrepancies))
    data_quality = _clamp(status_quality + verified_bonus - discrepancy_penalty)
    data_access_component = data_quality * 0.10

    missing_penalty = min(len(quality_check.missing_fields), 10) * 0.05
    suspicious_penalty = min(len(quality_check.suspicious_empty_fields), 10) * 0.03
    retry_penalty = 0.10 if quality_check.should_retry else 0.0
    qc_quality = _clamp(1.0 - missing_penalty - suspicious_penalty - retry_penalty)
    qc_component = qc_quality * 0.10

    raw = metadata_component + methods_component + results_component + data_access_component + qc_component
    # Track penalties for logging visibility.
    penalties = _clamp(1.0 - qc_quality, lo=0.0, hi=1.0)
    score = _clamp(raw, lo=0.0, hi=0.99)
    return ConfidenceBreakdown(
        score=score,
        metadata=metadata_component,
        methods=methods_component,
        results=results_component,
        data_access=data_access_component,
        penalties=_clamp(penalties + retry_penalty),
    )
