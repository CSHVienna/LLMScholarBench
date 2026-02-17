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
  2. Refusals ──────────────────┼────────────────────────────────────────────
     (batch_refusals)           │  A refusal is an explanation why the answer
                                │  is not provided. This step categorizes
                                │  refusals into four categories (contradictory
                                │  request, lack of information, unethical
                                │  request, other or no explanation).
                                │
                                ▼
  3. Author factuality ─────────┼────────────────────────────────────────────
     (batch_factuality)         │  Check that each mentioned author exists in
                                │  APS-OA. Filters to factually valid author
                                │  lists. Required before task factuality.
                                │
                                ▼
  4. Task-related factuality ───┼────────────────────────────────────────────
     (batch_factuality,         │  Check field, epoch, seniority (and optionally
      --task_name)              │  top_k, twins) against ground truth. One run
                                │  per task (field, epoch, seniority).
                                │
                                ▼
  5. GT demographics ───────────┼────────────────────────────────────────────
     (batch_disciplines)        │  Build ground-truth stats: gender and
                                │  ethnicity by discipline (APS-OA). Used for
                                │  demographics and parity analyses.
                                │
                                ▼
  6. Similarities ──────────────┼────────────────────────────────────────────
     (batch_similarities)       │  Co-authorship networks, demographics,
                                │  scholarly metrics, affiliations. Run per
                                │  task_name (e.g. top_k, field, epoch,
                                │  seniority, twins).
                                │
                                ▼
  7. Benchmarking ──────────────┼────────────────────────────────────────────
     (batch_benchmarking)       │  Aggregate metrics per attempt from
                                │  summaries, factuality, and similarities 
                                |  (e.g. validity, parity, factuality, 
                                |  similarity).
                                │
                                ▼
                    ┌───────────┴───────────┐
                    │  Results ready for    │
                    │  notebooks & plots    │
                    └───────────────────────┘
```

- **Valid responses:** Classify model outputs (valid / verbose / fixed / invalid / quota / error) and write per-model valid response sets and summaries.
- **Refusals:** A refusal is defined as an explanation why the answer is not provided. This step categorizes refusals into four categories: contradictory request, lack of information, unethical request, and other or no explanation.
- **Author factuality:** Verifies every recommended author exists in APS-OA; keep only responses with factually valid author lists.
- **Task-related factuality:** Check that field, epoch, seniority match ground truth for each valid author list (top_k, twins are not included as they are subjective).
- **GT demographics:** Compute discipline-level gender and ethnicity from APS-OA for benchmarking and fairness metrics.
- **Similarities:** Compute similarity and structure (co-authorship, demographics, metrics, affiliations) for valid responses per task.
- **Benchmarking:** Aggregates metrics per attempt from summaries, factuality, and similarities (e.g. validity, parity, factuality, similarity).

Note: Only `batch_disciplines.py` and `batch_refusals.py` are run once; the former for the entire ground-truth, and the latter for all experiment types (i.e., temperature and interventions). The rest of scripts must be run independently for each experiment type.

---

## How to run (pipeline)

You can run the pipeline **step by step** (from `Auditor/code/` with manual arguments) or use the **scripts** in `Auditor/scripts/`. Alternatively, you may **run all at once**, details below.

All batch scripts are run from the `code` directory with the library on `PYTHONPATH`.

### Step by step

0. **Set up the environment**

   ```bash
   cd code
   export PYTHONPATH="${PYTHONPATH}:."
   ```

   **Script:** The step scripts use `scripts/init.sh` (conda env + `PYTHONPATH`); run each `scripts/step*.sh` from `Auditor/`.

1. **Valid responses** — label validity (valid, verbose, fixed, invalid, quota, error)

   ```bash
   python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model gemma2-9b
   ```

   Or run for multiple models in parallel:

   ```bash
   parallel -j 6 python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b
   ```

   Use `--temperature_analysis` if results contain multiple temperatures per model.

   **Script:** `scripts/step1_validity.sh`

2. **Refusals** — detect refusals (explanations why the answer is not provided) and assign them to four categories (contradictory request, lack of information, unethical request, other or no explanation):

   ```bash
   python preprocessing/batch_refusals.py --summaries_dirs ../results/temperature_analysis/summaries ../results/interventions/summaries --output_dir ../results
   ```

   **Script:** `scripts/step2_refusals.sh`

3. **Author Factuality**

   ```bash
   python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b
   ```

   Or in parallel for all models:

   ```bash
   parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b
   ```

   **Script:** `scripts/step4_author_factuality.sh`

4. **Task factuality** (add `--task_name`) — run for each of `field`, `epoch`, `seniority` (order does not matter):

   ```bash
   python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b --task_name field
   ```

   Or in parallel over models and tasks:

   ```bash
   parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: field epoch seniority
   ```

   **Script:** `scripts/step5_task_factuality.sh`

5. **Gender and ethnicity by discipline (ground-truth)**

   ```bash
   python preprocessing/batch_disciplines.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --aps_data_zip ../../GTBuilder/APS/data/aps_20240130_2e52fdd7260ea462878821948a2a463ed9acb58a.zip --output ../results/ground_truth
   ```

   **Script:** `scripts/step3_demographics.sh`

6. **Similarities** (co-authorship, demographics, scholarly metrics, affiliations) — run for each task `top_k`, `field`, `epoch`, `seniority`, `twins`, `biased_top_k` (order does not matter):

   ```bash
   python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model gemma2-9b --task_name top_k
   ```

   Or in parallel over models and tasks:

   ```bash
   parallel -j 6 python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: top_k field epoch seniority twins biased_top_k
   ```

   **Script:** `scripts/step6_similarities.sh`

7. **Benchmarking** — aggregate metrics per attempt (validity, parity, factuality, similarity, etc.). Run per `--results_dir` (e.g. interventions or temperature_analysis), per `--model`, and per `--metric`:

   ```bash
   python preprocessing/batch_benchmarking.py --results_dir ../results/interventions --model gemma2-9b --metric validity --aps_oa_data_tar_gz ../../GTBuilder/APS/data/final_dataset.tar.gz --output_dir ../results
   ```

   Use `--temperature_analysis` when benchmarking temperature-analysis results. Add `--overwrite` to regenerate existing benchmark files.

   **Script:** `scripts/step7_benchmarking.sh`

### Run all at once

From the `Auditor/scripts` directory, run the full pipeline (steps 1–7 plus a final check) and write logs under a timestamped folder:

```bash
./runall.sh ../logs ../results
```

- **First argument** (`../logs`): directory where logs are written; each run creates `../logs/<YYYYMMDD_HHMMSS>/step1_validity.log`, …, `step8_check.log`.
- **Second argument** (`../results`): passed to `step8_check.sh` as the results root to verify expected file counts.
- **If step8_check.sh reports missing files:** use `Auditor/scripts/checker.sh` to see which ones are missing. Run it on each results subfolder (e.g. `./checker.sh ../results/benchmarks`, `./checker.sh ../results/factuality`, `../results/similarities`, `../results/summaries`, `../results/valid_responses`). The checker reports counts by model and by metric/task/kind; lines marked with ``expected N'' show missing or extra files.

Steps run in this order: 1 validity → 2 refusals → 3 demographics (GT) → 4 author factuality → 5 task factuality → 6 similarities → 7 benchmarking → 8 check. The step scripts use fixed data paths and model lists; edit the `scripts/step*.sh` files to match your setup.

---

## Plots and notebooks

All plotting and analysis is done in **Jupyter notebooks** under `notebooks/`.

- **`notebooks/base_audits/`** — **v1.0.0**: core validity, consistency, factuality, demographics, prestige, and similarity audits.
- **`notebooks/benchmark_interventions/`** — **v2.0.0**: ground truth, temperature, refusals, infrastructure, and intervention analyses (temperature, biased prompt, RAG, quadrants).

### benchmark_interventions (v2.0.0)

Notebooks live in `notebooks/benchmark_interventions/`.

| Notebook | Description |
|----------|-------------|
| `0_ground_truth.ipynb` | Ground-truth metadata and discipline-level demographics (from GT metadata step). |
| `1_refusal.ipynb` | Refusal analysis: when and how models refuse to answer. |
| `2_temperature.ipynb` | Effect of temperature on validity, factuality, and diversity. |
| `3_infrastructure.ipynb` | Model infrastructure: size, access (open/proprietary), reasoning vs non-reasoning, and related summaries. |
| `4_inter_temperature.ipynb` | Temperature intervention: comparing behavior across temperature settings. |
| `5_inter_biased_prompt.ipynb` | Biased-prompt intervention: effect of prompt framing on diversity and factuality. |
| `6_inter_rag.ipynb` | RAG intervention: effect of retrieval-augmented prompts on outputs. |
| `7_quadrants.ipynb` | Quadrant views: technical performance (validity, factuality, uniqueness) vs diversity/parity. |

### base_audits (v1.0.0)

Notebooks live in `notebooks/base_audits/`.

| Notebook | Description |
|----------|-------------|
| `1_basic_stats.ipynb` | Validity: response counts and validity flags (valid, verbose, fixed, invalid, quota, error) per model/task. |
| `2_consistency.ipynb` | Consistency of model outputs across attempts (e.g. same query, multiple runs). |
| `3_factuality.ipynb` | Factuality: author and task-level fact-check results and error rates. |
| `4_demographics.ipynb` | Demographics: gender and ethnicity representation in model outputs vs ground truth. |
| `5_prestige.ipynb` | Prestige: scholarly metrics and prestige-related statistics. |
| `6_similarity.ipynb` | Similarity: co-authorship and other similarity metrics from `batch_similarities`. |

