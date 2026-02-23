# GTBuilder — APS (American Physical Society) Ground Truth

This module builds ground-truth data for the LLMScholarBench benchmark by processing **APS (American Physical Society)** and **OpenAlex (OA)** data. It produces normalized entities (authors, affiliations, publications, rankings, demographics, etc.) used for evaluation.

**Inputs:** A combined APS–OpenAlex archive (`final_dataset.tar.gz`), optional raw APS CSV files, and (for demographics) an APS author-names file with gender labels.

**Outputs:** JSON/text and CSV artifacts under a configurable output directory (e.g. author lists, affiliations, rankings, demographics, coauthor networks).

---

## Scripts (`code/*.py`)

All batch scripts take CLI arguments (e.g. `--aps_os_data_tar_gz`, `--output_dir`). Run from the repo root or `GTBuilder/APS/code` with `code` on `PYTHONPATH` so that `libs` (and `libs.io`, `libs.constants`, etc.) resolve.

| Script | Description |
|--------|-------------|
| **`batch_affiliations.py`** | Reads institutions from the APS–OA tarball, assigns internal IDs, and writes an **affiliations** file (id_affiliation, openalex_id, display_name, type, country_code, ror, city). |
| **`batch_author_affiliations.py`** | Builds **author–affiliation** links: for each author, lists affiliations with OpenAlex institution IDs and the years they were associated. Uses authors, institutions, and author–institution–year tables from the tarball. |
| **`batch_authors.py`** | Extracts **authors** from the APS–OA authors table: assigns internal IDs, keeps display_name, created_date, updated_date, and writes the canonical author list. |
| **`batch_aps_authors_stats.py`** | Computes **APS-based author statistics** (works count, cited_by_count, h-index, i10-index, e-index, career age, years of activity, citations per paper per career year). Uses APS publications, authorships, and citations plus OA author mapping. With `--ranking`, also computes **APS metric rankings** (and percentiles) and writes a separate rankings file. Can write intermediate CSV (e.g. author stats) when not computing rankings. |
| **`batch_authors_demographics.py`** | End-to-end **demographics**: merges OA authors with APS author mapping and an APS gender file (e.g. author_names with gender_nq). Derives longest name, first/last name, then **ethnicity** (DemographicX on full name, Ethnicolor on last name; merged with a fallback) and **gender** (combined from NQ + ethnicity-aware assignment). Writes authors demographics (and optional intermediate CSVs). Single-process; for large runs use the parallel version. |
| **`batch_authors_demographics_parallel.py`** | Same demographics pipeline as `batch_authors_demographics.py` but runs **by task** (`--task ethnicity_dx`, `ethnicity_ec`, or `gender`) and uses **chunked parallel processing** for the gender step. Reads/writes intermediate CSVs (ethnicity_dx, ethnicity_ec, ethnicity, gender) so tasks can be re-run or resumed. |
| **`batch_coauthors_network.py`** | Builds **coauthor networks** per author: from authorships in the tarball, for each author computes the set of coauthors (other author internal IDs) and writes one record per author (id_network, id_author, openalex_id, aps_co_authors, collaboration_counts). |
| **`batch_disciplines.py`** | Reads **disciplines** from APS data (e.g. `APS_DISCIPLINES_FN` under `--aps_data_path`), assigns new IDs, and writes disciplines (new_id_discipline, aps_id_discipline, code, label). |
| **`batch_institutions_stats.py`** | From the APS–OA institutions table, builds **institution statistics** (h_index, mean_citedness_2yr, i10_index, cited_by_count, works_count) and writes them keyed by internal id_affiliation. |
| **`batch_missing_institutions.py`** | **Utility (no argparse):** reads a list of institution IDs from a file, calls the **OpenAlex API** for each, and saves the fetched institution data (e.g. summary_stats, geo) to a CSV. Used to backfill missing institutions (path and output file are set inside the script). |
| **`batch_publication_class.py`** | Joins APS–OA publications with the OA–APS publication mapping and APS **publication–topic** data (concept, area, discipline). Produces **publication classifications**: per publication, id_topic, id_topic_type, classification_type, primary flag, etc. |
| **`batch_publications.py`** | Builds the **publications** table from the APS–OA tarball: merges OA publications with the OA–APS mapping to attach aps_id_publication and aps_id_journal; outputs new_id_publication, oa/aps ids, doi, title, timestamps, language, oa_cited_by_count. |
| **`batch_user_rankings.py`** | **OpenAlex-based author rankings:** uses authors, authorships, publications, and citations from the tarball to compute e-index and career age, then **normalized citations per paper age**. Ranks authors by the configured metrics (e.g. citations, publications, h_index, e_index, citation_publication_age), computes percentiles, and writes **author rankings** (rr1_rank_*, rr2_rank_*, … and percentile fields). Also saves an intermediate CSV of author stats (APS_OA_AUTHORS_STATS_FN). |

### Shared library (`code/libs/`)

- **`io`** — Reading from tarball/CSV and writing JSON/CSV (paths, `constants` filenames).
- **`constants`** — Filenames (e.g. `APS_OA_AUTHORS_FN`, `APS_OA_INSTITUTIONS_FN`), ranking metric names, and small enums (e.g. gender/ethnicity labels).
- **`helpers`** — Utilities (e.g. `is_none`, `get_longest_name`).
- **`scholar`** — Citation metrics: `compute_h_index`, `compute_i10_index`, `compute_e_index`.
- **`ethnicity`** — Ethnicity prediction (DemographicX, Ethnicolor) and merging (e.g. `choose_ethnicity`).
- **`gender`** — Gender assignment from NQ + first name / ethnicity.
- **`parallel`** — Chunking and multiprocessing helpers used by `batch_authors_demographics_parallel`.

---

## Workflow

1. **Data:** Have `final_dataset.tar.gz` (APS–OA merged) and, where needed, raw APS CSVs and author_names (with gender) under a known path.
2. **Core entities:** Run `batch_authors`, `batch_affiliations`, `batch_publications`, `batch_disciplines` `batch_author_affiliations`, `batch_institutions_stats`.
3. **Author metrics:** Run `batch_aps_authors_stats` (with or without `--ranking`) and `batch_user_rankings` for OA-based rankings.
4. **Demographics:** Run `batch_authors_demographics` or `batch_authors_demographics_parallel` (with `--task` for parallel) after preparing the APS gender file.
5. **Networks & classification:** Run `batch_coauthors_network`, `batch_publication_class`.

For exact CLI options for each script, run e.g. `python batch_affiliations.py --help` (from a context where `libs` is importable).
