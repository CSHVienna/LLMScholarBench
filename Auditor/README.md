# Auditor

## Pipeline overview

The audit pipeline runs in sequence. Each step uses outputs from the previous one.

```
                     ┌─────────────────────┐
                     │   Raw experiment    │
                     │   responses (JSON)  │
                     └──────────┬──────────┘
                                │
                                ▼
  1. Valid responses ───────────┼────────────────────────────────────────────
     (batch_valid_answers)      │  Label each response: valid, verbose, fixed,
                                │  invalid, quota, error. Writes summaries and
                                │  valid_responses per model.
                                │
                                ▼
  2. Author factuality ─────────┼────────────────────────────────────────────
     (batch_factuality)         │  Check that each mentioned author exists in
                                │  APS-OA. Filters to factually valid author
                                │  lists. Required before task factuality.
                                │
                                ▼
  3. Task-related factuality ───┼────────────────────────────────────────────
     (batch_factuality,         │  Check field, epoch, seniority (and optionally
      --task_name)              │  top_k, twins) against ground truth. One run
                                │  per task (field, epoch, seniority).
                                │
                                ▼
  4. GT metadata ───────────────┼────────────────────────────────────────────
     (batch_disciplines)        │  Build ground-truth stats: gender and
                                │  ethnicity by discipline (APS-OA). Used for
                                │  demographics and parity analyses.
                                │
                                ▼
  5. Similarities ──────────────┼────────────────────────────────────────────
     (batch_similarities)       │  Co-authorship networks, demographics,
                                │  scholarly metrics, affiliations. Run per
                                │  task_name (e.g. top_k, field, epoch,
                                │  seniority, twins).
                                │
                                ▼
                    ┌───────────┴───────────┐
                    │  Results ready for    │
                    │  notebooks & plots    │
                    └───────────────────────┘
```

- **Valid responses:** Classify model outputs (valid / verbose / fixed / invalid / quota / error) and write per-model valid response sets and summaries.
- **Author factuality:** Verifies every recommended author exists in APS-OA; keep only responses with factually valid author lists.
- **Task-related factuality:** Check that field, epoch, seniority match ground truth for each valid author list (top_k, twins are not included as they are subjective).
- **GT metadata:** Compute discipline-level gender and ethnicity from APS-OA for benchmarking and fairness metrics.
- **Similarities:** Compute similarity and structure (co-authorship, demographics, metrics, affiliations) for valid responses per task.

---

## How to run (pipeline)

All batch scripts are run from the `code` directory with the library on `PYTHONPATH`.

1. **Set up the environment**

   ```bash
   cd code
   export PYTHONPATH="${PYTHONPATH}:."
   ```

2. **Valid responses** — label validity (valid, verbose, fixed, invalid, quota, error)

   ```bash
   python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model gemma2-9b
   ```

   Or run for multiple models in parallel:

   ```bash
   parallel -j 6 python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b
   ```

   Use `--temperature_analysis` if results contain multiple temperatures per model.

3. **Factuality**

   **3.1 Author factuality**:

   ```bash
   python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b
   ```

   Or in parallel for all models:

   ```bash
   parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b
   ```

   **3.2 Task factuality** (add `--task_name`): — run for each of `field`, `epoch`, `seniority` (order does not matter):

   ```bash
   python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b --task_name field
   ```

   Or in parallel over models and tasks:

   ```bash
   parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: field epoch seniority
   ```

4. **Disparities**

   **4.1 Gender and ethnicity by discipline (APS-OA):**

   ```bash
   python preprocessing/batch_disciplines.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --aps_data_zip ../data/aps_20240130_2e52fdd7260ea462878821948a2a463ed9acb58a.zip --output ../results
   ```

   **4.2 Similarities** (co-authorship, demographics, scholarly metrics, affiliations):

   ```bash
   python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model gemma2-9b --task_name top_k
   ```

   Or in parallel over models and tasks:

   ```bash
   parallel -j 6 python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: top_k field epoch seniority twins
   ```

---

## Plots and notebooks

All plotting and analysis is done in **Jupyter notebooks** under `notebooks/`.

- **`notebooks/basic_audits/`** — **v1.0.0**: core validity, consistency, factuality, demographics, prestige, and similarity audits.
- **`notebooks/benchmark_interventions/`** — **v2.0.0**: ground truth, temperature, refusals, infrastructure, and intervention analyses (temperature, biased prompt, RAG, quadrants).

### benchmark_interventions (v2.0.0)

Notebooks live in `notebooks/benchmark_interventions/`.

| Notebook | Description |
|----------|-------------|
| `0_ground_truth.ipynb` | Ground-truth metadata and discipline-level demographics (from GT metadata step). |
| `1_temperature.ipynb` | Effect of temperature on validity, factuality, and diversity. |
| `2_refusal.ipynb` | Refusal analysis: when and how models refuse to answer. |
| `3_infrastructure.ipynb` | Model infrastructure: size, access (open/proprietary), reasoning vs non-reasoning, and related summaries. |
| `4_inter_temperature.ipynb` | Temperature intervention: comparing behavior across temperature settings. |
| `5_inter_biased_prompt.ipynb` | Biased-prompt intervention: effect of prompt framing on diversity and factuality. |
| `6_inter_rag.ipynb` | RAG intervention: effect of retrieval-augmented prompts on outputs. |
| `7_quadrants.ipynb` | Quadrant views: technical performance (validity, factuality, uniqueness) vs diversity/parity. |

### basic_audits (v1.0.0)

| Notebook | Description |
|----------|-------------|
| `1_basic_stats.ipynb` | Validity: response counts and validity flags (valid, verbose, fixed, invalid, quota, error) per model/task. |
| `2_consistency.ipynb` | Consistency of model outputs across attempts (e.g. same query, multiple runs). |
| `3_factuality.ipynb` | Factuality: author and task-level fact-check results and error rates. |
| `4_demographics.ipynb` | Demographics: gender and ethnicity representation in model outputs vs ground truth. |
| `5_prestige.ipynb` | Prestige: scholarly metrics and prestige-related statistics. |
| `6_similarity.ipynb` | Similarity: co-authorship and other similarity metrics from `batch_similarities`. |

