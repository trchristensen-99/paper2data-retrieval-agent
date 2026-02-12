from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single quantitative or qualitative finding from the paper."""

    claim: str = Field(description="The factual claim, stripped of narrative spin")
    metric: str = Field(description="What was measured")
    value: str = Field(description="The numerical result")
    confidence_interval: Optional[str] = Field(default=None)
    p_value: Optional[str] = Field(default=None)
    effect_size: Optional[str] = Field(default=None)
    context: str = Field(description="Experimental context for this finding")
    confidence: float = Field(
        description="Agent confidence in extraction accuracy, 0-1", ge=0.0, le=1.0
    )


class FigureSummary(BaseModel):
    figure_id: str
    description: str
    key_findings: list[str]


class DataAccession(BaseModel):
    accession_id: str
    repository: str
    url: Optional[str] = None
    description: str
    is_accessible: Optional[bool] = None
    file_count: Optional[int] = None
    files_listed: Optional[list[str]] = None


class DataAvailabilityReport(BaseModel):
    overall_status: str
    claimed_repositories: list[str]
    verified_repositories: list[str]
    discrepancies: list[str]
    notes: str


class MethodsSummary(BaseModel):
    organisms: list[str]
    cell_types: list[str]
    assay_types: list[str]
    sample_sizes: dict[str, Any]
    statistical_tests: list[str]
    experimental_design: str
    methods_completeness: str = Field(
        description="Assessment of whether methods are detailed enough to reproduce"
    )


class MetadataRecord(BaseModel):
    title: str
    authors: list[str]
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    keywords: list[str] = []
    funding_sources: list[str] = []
    conflicts_of_interest: Optional[str] = None


class ResultsSummary(BaseModel):
    quantitative_findings: list[Finding]
    qualitative_findings: list[str]
    key_figures: list[FigureSummary] = []
    spin_assessment: str = Field(
        description="Brief note on whether author claims match the raw data"
    )


class PaperRecord(BaseModel):
    """Complete structured record for a scientific paper — the core database entry."""

    metadata: MetadataRecord
    methods: MethodsSummary
    results: ResultsSummary
    data_accessions: list[DataAccession]
    data_availability: DataAvailabilityReport
    code_repositories: list[str] = []
    extraction_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_version: str = "0.1.0"
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class SynthesisInput(BaseModel):
    metadata: MetadataRecord
    methods: MethodsSummary
    results: ResultsSummary
    data_accessions: list[DataAccession]
    data_availability: DataAvailabilityReport


class SynthesisOutput(BaseModel):
    record: PaperRecord
    retrieval_report_markdown: str
    retrieval_log_markdown: str


class RetryInstruction(BaseModel):
    agent_name: str
    reason: str


class QualityCheckOutput(BaseModel):
    missing_fields: list[str]
    suspicious_empty_fields: list[str]
    should_retry: bool
    retry_instructions: list[RetryInstruction] = []
    notes: str
