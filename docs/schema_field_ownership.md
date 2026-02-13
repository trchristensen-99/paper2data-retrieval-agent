# Paper2Data Schema Field Ownership

This document defines where each `PaperRecord` field is populated in the pipeline, and what can override it later.

## Pipeline Order

1. `anatomy_agent` (structure hints)
2. `metadata_agent`
3. `methods_agent`
4. `results_agent`
5. `data_availability_agent` (+ tool calls)
6. optional `quality_control_agent` retries
7. optional `metadata_enrichment_agent`
8. `synthesis_agent` (report/log generation)
9. manager canonical overwrite + confidence scoring

## Field Ownership (First Writer -> Overrides)

### `metadata.*`

- First: `metadata_agent`
- Possible overrides:
  - `metadata_enrichment_agent` (DOI/PMID/journal/publication date repair)
  - manager deterministic normalization:
    - journal/venue normalization
    - publication date normalization
    - keyword cleanup/repair
    - paper type fallback inference
    - placeholder stripping

### `methods.*`

- First: `methods_agent`
- Possible overrides:
  - manager deterministic cleanup:
    - placeholder stripping (e.g., `N/A`, sentinel tokens)
    - organism normalization is already applied in methods flow

### `results.*`

- First: `results_agent`
- Possible overrides:
  - manager `_backfill_dataset_results`:
    - populate/repair `dataset_profile`
    - populate fallback `dataset_properties`
    - parse/enrich `tables_extracted`
    - PRISMA normalization/backfill
    - `record_count` vs `source_corpus_size` consistency
    - dimension repairs (e.g., assessment type counts)

### `data_accessions[*]`

- First: `data_availability_agent`
- Possible overrides:
  - deterministic enrichment via tools (`check_url`, GEO/SRA, Figshare/Zenodo checks)
  - manager URL normalization/canonicalization (`fgshare`, `fle=`, malformed params)
  - manager file list merge + cleanup
  - classification filtering (external references removed from `data_accessions`)

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

### `code_repositories`

- First: manager deterministic derivation
  - code hosts (GitHub/GitLab/etc.)
  - code artifacts in repositories (e.g., Figshare package contents)
  - bundled code files
- Possible overrides:
  - dedupe/normalization only

### `related_resources`

- First: `data_availability_agent`
- Possible overrides:
  - manager pruning/dedupe:
    - remove duplicates against `code_repositories` and `data_accessions`
    - remove self-referential/noise links

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
- `data_availability`
- `code_repositories`
- `related_resources`

Synthesis still produces:

- `retrieval_report.md`
- `retrieval_log.md`

but does not get final authority over core structured fields.
