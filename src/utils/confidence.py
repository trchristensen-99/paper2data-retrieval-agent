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


def _overinterpretation_penalty(results: ResultsSummary, paper_type: str) -> float:
    """Estimate overinterpretation risk without exposing a separate spin field."""
    heavy_claim_terms = (
        "proves",
        "definitive",
        "guarantees",
        "always",
        "causes",
        "establishes causality",
    )
    text_blobs: list[str] = []
    text_blobs.extend([f.claim for f in results.experimental_findings])
    text_blobs.extend(results.synthesized_claims)
    joined = " ".join(text_blobs).lower()
    term_hits = sum(1 for term in heavy_claim_terms if term in joined)
    penalty = min(0.08, 0.02 * term_hits)

    if paper_type == "experimental" and results.experimental_findings:
        weak = 0
        for f in results.experimental_findings:
            has_support = bool((f.effect_size or "").strip() or (f.confidence_interval or "").strip() or (f.comparison or "").strip())
            if not has_support:
                weak += 1
        weak_ratio = weak / len(results.experimental_findings)
        if weak_ratio > 0.5:
            penalty += min(0.08, 0.12 * (weak_ratio - 0.5))
    return _clamp(penalty, 0.0, 0.16)


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
    # Explicit deterministic rubric with paper-type-aware criteria.
    paper_type = (metadata.paper_type or results.paper_type or "experimental").strip().lower()

    metadata_checks = [
        bool(metadata.title.strip()),
        bool(metadata.authors),
        bool((metadata.journal or "").strip()) and not (metadata.journal or "").strip().lower().startswith("unknown"),
        bool((metadata.publication_date or "").strip()) and not (metadata.publication_date or "").strip().lower().startswith("unknown"),
        bool(metadata.keywords),
        bool(metadata.doi or metadata.pmid),
        bool((metadata.paper_type or "").strip()),
    ]
    metadata_quality = sum(1.0 for ok in metadata_checks if ok) / len(metadata_checks)
    metadata_component = metadata_quality * 0.26

    methods_checks = [
        bool(methods.experimental_design.strip()),
        bool(methods.assay_types),
        bool(methods.sample_sizes),
        bool(methods.statistical_tests),
        bool(methods.organisms or methods.cell_types),
    ]
    methods_quality = sum(1.0 for ok in methods_checks if ok) / len(methods_checks)
    results_quality = 0.0
    if paper_type == "experimental":
        findings = results.experimental_findings
        if findings:
            finding_scores: list[float] = []
            for f in findings:
                checks = [
                    bool(f.claim.strip()),
                    bool(f.metric.strip()),
                    bool(f.value.strip()),
                    bool(f.context.strip()),
                    bool((f.effect_size or "").strip() or (f.confidence_interval or "").strip()),
                ]
                finding_scores.append(sum(1.0 for ok in checks if ok) / len(checks))
            results_quality = _clamp(sum(finding_scores) / len(finding_scores))
        results_weight = 0.30
        methods_weight = 0.22
    elif paper_type == "dataset_descriptor":
        props = results.dataset_properties
        prop_quality = 0.0
        if props:
            prop_quality = _clamp(
                sum(
                    1.0
                    for p in props
                    if p.property.strip() and p.value.strip() and p.context.strip()
                )
                / len(props)
            )
        license_bonus = 0.1 if (metadata.license or "").strip() else 0.0
        results_quality = _clamp(prop_quality + license_bonus)
        results_weight = 0.40
        methods_weight = 0.12
    elif paper_type in {"review", "meta_analysis"}:
        claims = results.synthesized_claims
        results_quality = _clamp(
            min(len(claims), 8) / 8.0 if claims else 0.0
        )
        results_weight = 0.34
        methods_weight = 0.18
    elif paper_type == "methods":
        benches = results.method_benchmarks
        bench_quality = _clamp(min(len(benches), 6) / 6.0 if benches else 0.0)
        results_quality = bench_quality
        results_weight = 0.28
        methods_weight = 0.24
    else:
        claims = results.synthesized_claims
        results_quality = _clamp(min(len(claims), 6) / 6.0 if claims else 0.0)
        results_weight = 0.38
        methods_weight = 0.14

    methods_component = methods_quality * methods_weight
    results_component = results_quality * results_weight

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
    data_access_component = data_quality * 0.14

    missing_penalty = min(len(quality_check.missing_fields), 10) * 0.05
    suspicious_penalty = min(len(quality_check.suspicious_empty_fields), 10) * 0.03
    retry_penalty = 0.10 if quality_check.should_retry else 0.0
    qc_quality = _clamp(1.0 - missing_penalty - suspicious_penalty - retry_penalty)
    qc_component = qc_quality * 0.08
    overclaim_penalty = _overinterpretation_penalty(results, paper_type)

    raw = metadata_component + methods_component + results_component + data_access_component + qc_component - overclaim_penalty
    # Track penalties for logging visibility.
    penalties = _clamp(1.0 - qc_quality, lo=0.0, hi=1.0)
    score = _clamp(raw, lo=0.0, hi=0.99)
    return ConfidenceBreakdown(
        score=score,
        metadata=metadata_component,
        methods=methods_component,
        results=results_component,
        data_access=data_access_component,
        penalties=_clamp(penalties + retry_penalty + overclaim_penalty),
    )
