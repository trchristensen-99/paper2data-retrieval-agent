from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


PAPER_TYPES = {
    "experimental",
    "dataset_descriptor",
    "review",
    "methods",
    "meta_analysis",
    "commentary",
}


class ExperimentalFinding(BaseModel):
    """A single quantitative result for experimental-style reporting."""

    claim: str = Field(description="The factual claim, stripped of narrative spin")
    metric: str = Field(description="What was measured")
    value: str = Field(description="The numerical result")
    confidence_interval: Optional[str] = Field(default=None)
    p_value: Optional[str] = Field(default=None)
    effect_size: Optional[str] = Field(default=None)
    comparison: Optional[str] = Field(default=None)
    context: str = Field(description="Experimental context for this finding")
    confidence: float = Field(
        description="Agent confidence in extraction accuracy, 0-1", ge=0.0, le=1.0
    )


class Finding(ExperimentalFinding):
    """Backward-compatible alias for older code paths."""


class FigureSummary(BaseModel):
    figure_id: str
    description: str
    key_findings: list[str]


class DescriptiveStat(BaseModel):
    property: str = Field(description="Dataset/review descriptive property name")
    value: str = Field(description="Value for this descriptive property")
    context: str = Field(description="Context/notes for the property")


class MethodBenchmark(BaseModel):
    task: str
    metric: str
    value: str
    baseline: Optional[str] = None
    context: str


class DatasetColumn(BaseModel):
    name: str
    category: Optional[str] = None
    description: Optional[str] = None


class PrismaFlow(BaseModel):
    database_records_total: Optional[int] = None
    citation_review_records: Optional[int] = None
    expert_records: Optional[int] = None
    records_identified_total: Optional[int] = None
    duplicates_removed: Optional[int] = None
    records_after_duplicate_removal: Optional[int] = None
    screened: Optional[int] = None
    excluded_title_abstract: Optional[int] = None
    full_text_review: Optional[int] = None
    excluded_full_text: Optional[int] = None
    included: Optional[int] = None

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_int(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            text = str(value).strip()
            digits = "".join(ch for ch in text if ch.isdigit())
            return int(digits) if digits else None
        except Exception:  # noqa: BLE001
            return None

    def as_compact_dict(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, int):
                out[key] = value
        return out


class DatasetProfile(BaseModel):
    name: Optional[str] = None
    format: list[str] = []
    record_count: Optional[int] = None
    data_point_count: Optional[int] = None
    columns: Optional[int] = None
    temporal_coverage: Optional[str] = None
    source_corpus_size: Optional[int] = None
    dimensions: dict[str, Any] = {}
    physical_dimensions: dict[str, Any] = {}
    conceptual_dimensions: dict[str, Any] = {}
    version: Optional[str] = None
    license: Optional[str] = None
    repository_contents: list[str] = []
    prisma_flow: PrismaFlow = Field(default_factory=PrismaFlow)
    processing_pipeline_summary: Optional[str] = None
    column_schema: list[DatasetColumn] = []

    @field_validator("prisma_flow", mode="before")
    @classmethod
    def _normalize_prisma_flow(cls, value):
        alias_map = {
            "records_identified": "database_records_total",
            "records_screened": "screened",
            "full_text_reviews": "full_text_review",
            "full_text_assessed": "full_text_review",
            "studies_included": "included",
        }
        if isinstance(value, PrismaFlow):
            return value
        if not isinstance(value, dict):
            return PrismaFlow()
        normalized: dict[str, Any] = {}
        for key, raw in value.items():
            k = alias_map.get(str(key).strip(), str(key).strip())
            normalized[k] = raw
        return PrismaFlow.model_validate(normalized)


class ExtractedTable(BaseModel):
    table_id: str
    title: Optional[str] = None
    columns: list[str] = []
    summary: Optional[str] = None
    key_content: list[str] = []


class RelatedResource(BaseModel):
    name: str
    url: Optional[str] = None
    type: str = Field(description="visualization|tool|standard|related_dataset")
    description: Optional[str] = None


class DataAccession(BaseModel):
    accession_id: str
    category: str = Field(
        default="external_reference",
        description="primary_dataset | supplementary_data | external_reference",
    )
    repository: str
    url: Optional[str] = None
    description: str
    data_format: Optional[str] = None
    is_accessible: Optional[bool] = None
    file_count: Optional[int] = None
    files_listed: Optional[list[str]] = None
    total_size_bytes: Optional[int] = None
    estimated_download_seconds: Optional[float] = None
    download_probe_url: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def _norm_category(cls, value):
        v = str(value or "").strip().lower()
        allowed = {"primary_dataset", "supplementary_data", "external_reference"}
        return v if v in allowed else "external_reference"


class DataAvailabilityReport(BaseModel):
    overall_status: str
    claimed_repositories: list[str]
    verified_repositories: list[str]
    discrepancies: list[str]
    notes: str
    check_status: str = "ok"

    @field_validator("claimed_repositories", "verified_repositories", "discrepancies", mode="before")
    @classmethod
    def _none_to_list(cls, value):
        return [] if value is None else value


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

    @field_validator("organisms", "cell_types", "assay_types", "statistical_tests", mode="before")
    @classmethod
    def _none_to_list(cls, value):
        return [] if value is None else value

    @field_validator("sample_sizes", mode="before")
    @classmethod
    def _none_to_dict(cls, value):
        return {} if value is None else value


class MetadataRecord(BaseModel):
    title: str
    authors: list[str]
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    publication_status: Optional[str] = None
    keywords: list[str] = []
    funding_sources: list[str] = []
    conflicts_of_interest: Optional[str] = None
    paper_type: Optional[str] = Field(
        default=None,
        description="experimental|dataset_descriptor|review|methods|meta_analysis|commentary",
    )
    license: Optional[str] = None
    category: Optional[str] = Field(
        default=None,
        description="Fixed top-level field (e.g., biology, chemistry, physics, computer_science, mathematics_statistics, social_science, interdisciplinary)",
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Fixed subfield within the selected field",
    )

    @field_validator("authors", "keywords", "funding_sources", mode="before")
    @classmethod
    def _none_to_list(cls, value):
        return [] if value is None else value

    @field_validator("paper_type", mode="before")
    @classmethod
    def _norm_paper_type(cls, value):
        v = str(value or "").strip().lower()
        return v if v in PAPER_TYPES else None


class ResultsSummary(BaseModel):
    paper_type: Optional[str] = None
    experimental_findings: list[ExperimentalFinding] = []
    dataset_properties: list[DescriptiveStat] = []
    dataset_profile: Optional[DatasetProfile] = None
    synthesized_claims: list[str] = []
    method_benchmarks: list[MethodBenchmark] = []
    key_figures: list[FigureSummary] = []
    tables_extracted: list[ExtractedTable] = []

    @field_validator(
        "experimental_findings",
        "dataset_properties",
        "tables_extracted",
        "synthesized_claims",
        "method_benchmarks",
        "key_figures",
        mode="before",
    )
    @classmethod
    def _none_to_list(cls, value):
        return [] if value is None else value

    @field_validator("paper_type", mode="before")
    @classmethod
    def _norm_result_paper_type(cls, value):
        v = str(value or "").strip().lower()
        return v if v in PAPER_TYPES else None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_result_shape(cls, data):
        if not isinstance(data, dict):
            return data
        if "experimental_findings" not in data and "quantitative_findings" in data:
            data["experimental_findings"] = data.get("quantitative_findings")
        if "synthesized_claims" not in data and "qualitative_findings" in data:
            data["synthesized_claims"] = data.get("qualitative_findings")
        if "paper_type" not in data:
            data["paper_type"] = None
        data.setdefault("dataset_profile", None)
        data.setdefault("dataset_properties", [])
        data.setdefault("method_benchmarks", [])
        data.setdefault("tables_extracted", [])
        return data


class PaperRecord(BaseModel):
    """Complete structured record for a scientific paper — the core database entry."""

    metadata: MetadataRecord
    methods: MethodsSummary
    results: ResultsSummary
    data_accessions: list[DataAccession]
    data_availability: DataAvailabilityReport
    code_repositories: list[str] = []
    vcs_repositories: list[str] = []
    archival_repositories: list[str] = []
    code_available: Optional[bool] = None
    related_resources: list[RelatedResource] = []
    extraction_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_version: str = "0.1.0"
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extraction_confidence_breakdown: dict[str, Any] = {}

    @field_validator(
        "data_accessions",
        "code_repositories",
        "vcs_repositories",
        "archival_repositories",
        "related_resources",
        mode="before",
    )
    @classmethod
    def _none_to_list(cls, value):
        return [] if value is None else value


class SynthesisInput(BaseModel):
    metadata: MetadataRecord
    methods: MethodsSummary
    results: ResultsSummary
    data_accessions: list[DataAccession]
    data_availability: DataAvailabilityReport
    code_repositories: list[str] = []
    vcs_repositories: list[str] = []
    archival_repositories: list[str] = []
    code_available: Optional[bool] = None
    related_resources: list[RelatedResource] = []


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


class MetadataEnrichmentOutput(BaseModel):
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    notes: str


class PaperAnatomyOutput(BaseModel):
    sections: list[str] = []
    tables: list[str] = []
    figures: list[str] = []
    urls: list[str] = []
    accession_candidates: list[str] = []
    prisma_flow: dict[str, int] = {}
    notes: str = ""
