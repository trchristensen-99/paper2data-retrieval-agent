# Paper2Data Schema Field Ownership

This document defines where each `PaperRecord` field is populated in the pipeline, and what can override it later.

## Pipeline Order

1. Manager text pre-cleaning
   - HTML entity decode
   - whitespace normalization
2. `anatomy_agent` (structure hints)
3. `metadata_agent`
4. `methods_agent`
5. `results_agent`
6. `data_availability_agent` (+ tool calls)
7. optional `quality_control_agent` retries
8. optional `metadata_enrichment_agent`
9. manager deterministic post-processing
   - URL normalization/canonicalization
   - table topology normalization + de-duplication
   - dataset/profile backfills
   - PRISMA normalization
   - sample-size vs findings reclassification
   - data-asset derivation from accession files
   - keyword/method placeholder cleanup
   - ontology-grounded assay mapping normalization
   - exact-substring provenance normalization
   - code/resource classification
   - file-list normalization
10. `synthesis_agent` (report/log generation)
11. manager canonical overwrite + confidence scoring

## Field Ownership (First Writer -> Overrides)

### `metadata.*`

- First: `metadata_agent`
- Possible overrides:
  - `metadata_enrichment_agent` (DOI/PMID/journal/publication date repair)
  - manager deterministic normalization:
    - journal/venue normalization
    - publication date normalization
    - publication status assignment (`confirmed` / `advance_access`)
    - license normalization (`CC-BY-4.0`, `CC0-1.0` when detectable)
    - keyword cleanup/repair
    - paper type fallback inference
    - placeholder stripping

### `methods.*`

- First: `methods_agent`
- Possible overrides:
  - manager deterministic cleanup:
    - placeholder stripping (e.g., `N/A`, sentinel tokens)
    - organism normalization is already applied in methods flow
    - domain-profile normalization (bio vs meta-research vs computational)
    - `experimental_design_steps` atomization (chronological, count-aware steps)
    - `assay_type_mappings` ontology-style normalization (`raw`, `mapped_term`, `ontology_id`, `vocabulary`)
    - hard coercion of malformed model outputs (string steps -> structured step objects)

### `results.*`

- First: `results_agent`
- Possible overrides:
  - manager `_backfill_dataset_results`:
    - populate/repair `dataset_profile`
    - populate fallback `dataset_properties`
    - parse/enrich `tables_extracted`
      - row-level `data` objects (SQL-ingestible)
      - structural normalization of section-break rows into explicit `category` column where needed
      - duplicate table variant suppression (`Table N` dropped when normalized `Table N-part*` exists)
      - per-table provenance hints
    - PRISMA normalization/backfill via typed `PrismaFlow`
    - `record_count` vs `source_corpus_size` consistency
    - dimension repairs (e.g., assessment type counts)
    - split dimensions into `physical_dimensions` and `conceptual_dimensions`
    - column schema cleanup (null/ghost filtering)
    - quantitative finding backfill for key percentage/stat claims when extractor output is sparse
    - fallback finding derivation from numeric dataset properties
    - strict partition rule between `dataset_properties` and `experimental_findings`:
      - static artifact descriptors (rows, columns, bytes, file/record counts, version/license/format) stay in `dataset_properties`
      - scientific outcome statements (percentages, p-values, effect-style results) stay in `experimental_findings`
      - dedupe pass removes duplicate metric/value pairs across both collections
    - reclassification of resource/input counts into `methods.sample_sizes`
    - per-item provenance defaults for findings/properties/tables

### `data_accessions[*]`

- First: `data_availability_agent`
- Possible overrides:
  - deterministic enrichment via tools (`check_url`, GEO/SRA, Figshare/Zenodo checks)
    - URL repair retry path (OCR typo fixes, query parameter fixes)
    - `url_repaired=true` when repair path changed final URL
  - manager URL normalization/canonicalization (`fgshare`, `fle=`, malformed params)
  - manager file list merge + cleanup (`P2D_STRICT_FILE_LIST` optional strict mode)
  - classification filtering (external references removed from `data_accessions`)
  - canonical figshare URL construction when article/file ids are known
  - identifier normalization for joins:
    - `system` (`DOI`, `GEO`, `SRA`, `OTHER`)
    - `normalized_id` (e.g., `doi:...`, `geo:GSE...`, `sra:SRR...`)
    - `repository_type` (`generalist`, `domain_specific`, `other`)

### `data_assets[*]`

- First/final: manager deterministic derivation from `data_accessions[*].files_listed`, accession URLs, and dataset profile hints.
- Includes:
  - `content_type` classification (e.g., `gene_disease_associations`, `tabular_dataset`, `analysis_code`)
  - file-level URL linkage (`download_probe_url`/accession URL)
  - optional row-count hints and provenance

### `data_availability.*`

- First: `data_availability_agent`
- Possible overrides:
  - deterministic recompute from enriched accessions:
    - `claimed_repositories`
    - `verified_repositories`
    - `overall_status`
    - `check_status`
  - contradiction guard:
    - unresolved verification discrepancies can downgrade status from `accessible`
  - fallback guard:
    - fallback path cannot return confident `accessible` if checks failed/unverified

### `code_repositories`, `vcs_repositories`, `archival_repositories`, `code_available`

- First: manager deterministic derivation/classification
  - `vcs_repositories`: Git/SVN-style code hosts
  - `archival_repositories`: Figshare/Zenodo/bundled-file code artifacts
  - `code_repositories`: canonical VCS-facing list for UI/queries
  - `code_available`: boolean convenience flag
- Possible overrides: dedupe/normalization only

### `related_resources`

- First: `data_availability_agent`
- Possible overrides:
  - manager pruning/dedupe:
    - remove duplicates against code/data fields
    - remove self-referential/noise links
    - suppress generic host-only links when deep target is unresolved

### `extraction_confidence`

- First/final: manager deterministic rubric (`compute_extraction_confidence`)

### `extraction_confidence_breakdown`

- First/final: manager (`confidence.as_dict()`)
- Contains per-section:
  - `weighted`
  - `quality`
  - `weight`
  - plus penalty summary

### `extraction_timestamp`, `agent_version`

- First: synthesis path (`run_synthesis_agent` / `fallback_synthesis`)
- Usually retained in final record.

## Canonical Final Record Guard

After synthesis returns, manager force-overwrites structured record sections with canonical pipeline outputs to prevent synthesis-model drift:

- `metadata`
- `methods`
- `results`
- `data_accessions`
- `data_assets`
- `data_availability`
- `code_repositories`
- `vcs_repositories`
- `archival_repositories`
- `code_available`
- `related_resources`

Synthesis still produces:

- `retrieval_report.md`
- `retrieval_log.md`

but does not get final authority over core structured fields.

## Model Defaults

- Global default model: `gpt-5-mini` (`P2D_DEFAULT_MODEL`)
- Per-agent overrides available via `P2D_MODEL_*` variables.
