# LLMScholarBench — Data for Visualization Tool

This document describes the **result types**, **experiments**, and **file layout** of the audit data produced for the visualization tool. It explains how to use each file and how to build leaderboards and analyses from the benchmarks and audit outputs.

---

## 1. Result types: baseline vs interventions

We have two broad result types:

- **Baseline** — Queries to the LLMs **without** any intervention that might influence outputs. Only the regular/template prompt per task is used. This is the reference condition for comparison.
- **Interventions** — Queries where we deliberately change the setup:
  - **Varying temperature** — multiple temperature values across LLMs and tasks to see if and how benchmarks change.
  - **Constrained prompting** — the prompt is changed to request more people from a given group (e.g. diversity-oriented framing).
  - **RAG (retrieval-augmented generation)** — the model is allowed to query the web; we compare results with and without this capability.

All benchmark scores (and related factuality/similarity files) are tagged by experiment and task so you can filter baseline vs each intervention.

---

## 2. Directory and file layout

The vistool data is organized as follows (paths are relative to the vistool root, e.g. `vistool/` or `RESULTS_DIR/vistool/`):

```
vistool/
├── audit/
│   ├── interventions/          # All intervention runs (temperature, biased prompt, RAG)
│   │   ├── benchmarks.csv
│   │   ├── factuality_author.csv
│   │   ├── factuality_epoch.csv
│   │   ├── factuality_field.csv
│   │   ├── factuality_seniority.csv
│   │   ├── similarities_top_k.csv
│   │   ├── similarities_epoch.csv
│   │   ├── similarities_field.csv
│   │   ├── similarities_seniority.csv
│   │   └── similarities_twins.csv
│   ├── temperature_analysis/   # Temperature sweep (multiple temps per model/task)
│   │   ├── benchmarks.csv
│   │   └── (same factuality_* and similarities_* files as above)
│   └── refusals.csv            # All detected refusals (single file for both experiments)
├── extra/                      # Supplementary datasets
│   ├── disciplines_demographics.csv
│   └── nobel-prize-laureates.csv
└── ground_truth/               # APS author metadata for networks and PCA
    ├── coauthorships_edgelist.txt
    ├── authors_demographics.csv
    ├── authors_stats.csv
    ├── authors_PCA.csv
    └── summary_PCA.csv
```

---

## 3. Benchmarks and leaderboards

The **`benchmarks.csv`** files contain **one row per (model, task, metric, …)** with the score for that metric. They are the main source for building leaderboards.

- **Columns** (typical): `model_access`, `model_size`, `model_class`, `model`, `grounded`, `temperature`, `date`, `time`, `task_name`, `task_param`, `metric_name`, `metric_value`.
- **Location**:
  - `audit/interventions/benchmarks.csv` — all intervention runs (baseline + temperature + biased prompt + RAG).
  - `audit/temperature_analysis/benchmarks.csv` — temperature-sweep runs (multiple temperatures per model/task).

**How to build leaderboards and run analyses:**

- Filter rows by `task_name`, `task_param`, `model`, `grounded`, and optionally `temperature`, then aggregate or rank by `metric_name` / `metric_value` as needed.

---

## 4. Baseline (no interventions)

- **Definition:** Queries with the standard prompt per task, no temperature sweep, no constrained prompt, no RAG.
- **Setup:** 22 LLMs, 5 tasks (**top_k**, **field**, **epoch**, **seniority**, **twins**), one fixed temperature per LLM (chosen for highest accuracy).
- **Data source:** `audit/interventions/benchmarks.csv`.
- **Filters to get baseline only:**
  - Exclude **Gemini grounded** models (e.g. `grounded == True` or model name containing “grounded”).
  - Exclude **biased_top_k** task: keep only the standard tasks (e.g. `task_name` in `['top_k', 'field', 'epoch', 'seniority', 'twins']`).

Use this subset to build the **main baseline leaderboard** by model and metric.

---

## 5. Intervention: varying temperature

- **Goal:** See whether and how benchmark scores change when temperature varies across LLMs and tasks.
- **Data source:** `audit/temperature_analysis/benchmarks.csv`.
- **Filters:** Same as baseline for comparability:
  - Exclude **Gemini grounded** models.
  - Exclude **biased_top_k** task.

You can compare, per (model, task, metric), scores across different `temperature` values to study sensitivity to temperature.

---

## 6. Intervention: constrained prompting (biased prompt)

- **Goal:** Compare outputs when the prompt explicitly asks for more people from a given group vs the standard prompt.
- **Data source:** `audit/interventions/benchmarks.csv`.
- **Intervention subset:** `task_name == 'biased_top_k'`.
- **Baseline subset:** standard top-k with `task_param == 'top_100'` (same task type, no diversity-oriented prompt).

Compare metrics between the biased-prompt condition and the top_100 baseline to measure the effect of constrained prompting.

---

## 7. Intervention: RAG (web search)

- **Goal:** Compare outputs when the model can use web search (RAG) vs when it cannot.
- **Data source:** `audit/interventions/benchmarks.csv`.
- **Intervention subset:** **Gemini grounded** models (e.g. `gemini-2.5-flash-grounded`, `gemini-2.5-pro-grounded`).
- **Baseline subset:** **Gemini non-grounded** models (e.g. `gemini-2.5-flash`, `gemini-2.5-pro`).

Compare the same metrics between grounded (RAG) and non-grounded (no RAG) to quantify the impact of RAG.

---

## 8. Factuality and recommendations (per request)

To inspect **which authors were recommended** and **how they were checked** against ground truth, use the factuality CSVs (and optionally the similarity CSVs).

- **Path pattern:**  
  `audit/{interventions|temperature_analysis}/factuality_<task>.csv`  
  where `<task>` is one of: **author**, **epoch**, **field**, **seniority**.

| File | Content |
|------|--------|
| `factuality_author.csv` | Which recommended authors are **real** (matched to APS/OpenAlex). |
| `factuality_epoch.csv` | Whether recommended authors match the **requested time period** (epoch). |
| `factuality_field.csv` | Whether recommended authors match the **requested field**. |
| `factuality_seniority.csv` | Whether recommended authors match the **requested seniority** (e.g. early-career vs senior). |

Use these to drill down from a leaderboard (model/task/run) to the actual recommendations and their factuality flags.

---

## 9. Similarity (within recommended sets)

Similarity metrics describe how similar the recommended authors are to each other (e.g. demographics, co-authorship, scholarly metrics). Many of these are also aggregated into `benchmarks.csv` for leaderboard use.

- **Path pattern:**  
  `audit/{interventions|temperature_analysis}/similarities_<task>.csv`  
  with `<task>` in: **top_k**, **epoch**, **field**, **seniority**, **twins**.

These files give per-request similarity along multiple dimensions (e.g. gender, ethnicity, scholarly similarity, co-authorship). Use them for detailed diversity and structure analysis beyond the single metrics in the benchmarks file.

---

## 10. Refusals

- **Path:** `audit/refusals.csv`.
- **Content:** Requests where the model **explicitly refused** to answer (e.g. declined to give recommendations). Includes model, task, and refusal cluster/label.
- **Use:** Analyze when and how often models refuse, and how refusals vary by task or intervention.

---

## 11. Ground truth and extra data

- **`ground_truth/`** — Information about **American Physical Society (APS)** authors used as reference:
  - **Coauthorship:** e.g. `coauthorships_edgelist.txt` to build a **coauthorship network**.
  - **PCA / 2D visualization:** e.g. `authors_PCA.csv` and `summary_PCA.csv` from a PCA model on bibliometrics to place authors in a 2D space.
  - **Demographics and stats:** e.g. `authors_demographics.csv`, `authors_stats.csv` for metadata and bibliometric indicators.

- **`extra/`** — Additional reference data, e.g.:
  - `disciplines_demographics.csv` — discipline-level demographics.
  - `nobel-prize-laureates.csv` — list of Nobel laureates (e.g. for prestige or name matching).

---

## 12. Quick reference: which file for what

| Goal | Primary file(s) | Filters / notes |
|------|------------------|------------------|
| Baseline leaderboard (22 LLMs, 5 tasks) | `interventions/benchmarks.csv` | Exclude Gemini grounded; exclude `task_name == 'biased_top_k'` |
| Temperature sensitivity | `temperature_analysis/benchmarks.csv` | Exclude Gemini grounded; exclude biased_top_k |
| Constrained-prompt effect | `interventions/benchmarks.csv` | Intervention: `task_name == 'biased_top_k'`; baseline: `task_param == 'top_100'` |
| RAG vs no-RAG | `interventions/benchmarks.csv` | Intervention: Gemini grounded; baseline: Gemini non-grounded |
| Who was recommended / real authors | `factuality_author.csv` | By model, task, run |
| Epoch/field/seniority match | `factuality_epoch.csv`, `factuality_field.csv`, `factuality_seniority.csv` | By model, task, run |
| Similarity within recommendations | `similarities_*.csv` | By task (top_k, epoch, field, seniority, twins) |
| When models refused | `refusals.csv` | By model, task, cluster |
| Coauthorship network / PCA 2D | `ground_truth/coauthorships_edgelist.txt`, `ground_truth/authors_PCA.csv` | APS authors |

This README and the file layout align with the notebook `notebooks/7_vistool.ipynb`, which generates the vistool audit outputs from the raw pipeline results.
