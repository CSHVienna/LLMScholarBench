# GTBuilder — Ground Truth for Auditing

GTBuilder allocates modules to generate **standardized ground-truth files** used later in the LLMScholarBench **auditing** pipeline. These files define canonical entities (authors, publications, affiliations, citations, rankings, demographics, etc.) against which model outputs are evaluated.

---

## Current data: American Physical Society (APS)

At present, ground truth is built from **publications** (and associated authors, citations, journals, etc.) of the **American Physical Society (APS)**. The workflow is split into two complementary parts:

| Component | Role |
|-----------|------|
| **APS** | Extracts and normalizes data **purely from APS** sources. Produces core entities (authors, publications, affiliations, disciplines, author–affiliation links, coauthor networks, publication classifications) and APS-based author statistics and rankings. |
| **OpenAlex_API_Pipeline** | **Augments** the APS data with information from **OpenAlex**. It enriches the corpus with **global indicators** (e.g. citation counts, summary stats, works counts at the author and institution level) and **alternative author names** — i.e. all the ways each author has written their name across papers in OpenAlex (display name and display name alternatives). This supports matching and evaluation when model outputs use different name variants. |

The combined APS + OpenAlex data is then used by the APS module to produce the final standardized ground-truth artifacts consumed by the Auditor.

---

## Structure

- **`APS/`** — Scripts and workflows to build ground truth from APS (and merged APS–OpenAlex) data. See [APS/README.md](APS/README.md) for batch scripts, libraries, and workflow.
- **`OpenAlex_API_Pipeline/`** — Pipeline to retrieve publication, author, and institution data from the OpenAlex API to augment APS. See [OpenAlex_API_Pipeline/README.md](OpenAlex_API_Pipeline/README.md) for usage and data layout.
