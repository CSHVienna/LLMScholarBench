# LLMScholarBench

[![Code DOI](https://img.shields.io/badge/DOI-Code-blue)](https://doi.org/10.5281/zenodo.20415692)
[![Data DOI](https://img.shields.io/badge/DOI-Data-green)](https://doi.org/10.5281/zenodo.20417107)

**LLMScholarBench** is a benchmark for auditing large language models (LLMs) in LLM-based scholar recommendations. It provides a framework to collect model outputs, standardize ground-truth data, and analyze recommendations along dimensions such as consistency, factuality, demographics, prestige, and similarity.

## Requirements

- **Python**: 3.11.11 (recommended; use [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/) to manage versions)

```bash
# Example with pyenv
pyenv install 3.11.11
pyenv local 3.11.11
```

## Repository structure

The repository is organized into three main modules:

### 1. LLMCaller

Code to **query LLMs** and collect scholar-recommendation responses. It supports multiple providers (e.g. OpenRouter, Gemini, Groq), configurable prompts and schemas, and structured storage of outputs for downstream auditing.

- **Location**: `LLMCaller/`
- **Features**: Async execution across many models, JSON schema validation, experiment categories (top-k experts, field/epoch/seniority, twins), and configurable temperatures.
- **Setup**: See `LLMCaller/README.md` for credentials, `config/paths.json`, and installation (e.g. `pip install -r LLMCaller/requirements.txt`).

### 2. Auditor

Code to **analyze LLM outputs** against ground truth. It runs preprocessing pipelines (valid answers, factuality, similarities) and provides notebooks for consistency, factuality, demographics, prestige, and similarity analyses.

- **Location**: `Auditor/`
- **Features**: Batch preprocessing scripts, factuality checks (author, field, epoch, seniority), demographic and similarity metrics, and visualization notebooks.
- **Setup**: Set `PYTHONPATH` from `Auditor/code/` and use the batch scripts; see `Auditor/README.md` for the full pipeline.

### 3. GTBuilder

Module to **prepare ground-truth data** for downstream analyses. Currently implemented for **American Physical Society (APS)** publications (physics domain), with OpenAlex used for author/institution metadata.

- **Location**: `GTBuilder/`
- **Contents**:
  - **APS**: Scripts to build author stats, demographics, affiliations, coauthorship networks, disciplines, and publication classes from APS + OpenAlex.
  - **OpenAlex API Pipeline**: Scripts to fetch publications, authors, and institutions from the OpenAlex API (used by the APS pipeline).

## Installation

1. Clone the repository and use Python 3.11.11:

   ```bash
   git clone https://github.com/CSHVienna/LLMScholarBench.git
   cd LLMScholarBench
   # Use Python 3.11.11 (e.g. pyenv local 3.11.11)
   ```

2. Install dependencies for the modules you plan to use (each has its own `requirements.txt`):

   ```bash
   pip install -r LLMCaller/requirements.txt
   pip install -r Auditor/requirements.txt
   pip install -r GTBuilder/APS/requirements.txt
   pip install -r GTBuilder/OpenAlex_API_Pipeline/requirements.txt
   ```
   Install only the lines for the modules you need.

3. Configure and run each module according to its README (`LLMCaller/`, `Auditor/`, `GTBuilder/*/`).



## Related papers

| Version | Description | Link |
|--------|-------------|------|
| **v2.0** | Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation | [arXiv:2602.08873](https://arxiv.org/abs/2602.08873) (KDD'26) |
| **v1.0** | Whose Name Comes Up? I:  Auditing LLM-Based Scholar Recommendations | [arXiv:2506.00074](https://arxiv.org/abs/2506.00074) *(under review)* |



## Citation

If you use LLMScholarBench in your work, please cite the benchmark and/or the relevant paper(s) below.

**BibTeX (v2.0.0):**

```bibtex
@article{espinnoboa2026llmscholarbench,
  author      = {Esp\'{i}n-Noboa, Lisette and M\'{e}ndez, Gonzalo G.},
  title       = {Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation},
  booktitle   = {The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year        = {2026},
  publisher   = {Association for Computing Machinery},
  address     = {Jeju, Korea},
  doi         = {10.1145/3770855.3817543},
  note        = {v2.0.0}
}
```

**BibTeX (v1.0.0):**

```bibtex
@article{barolo2025llmscholaraudits,
  author    = {Barolo, Daniele and Valentin, Chiara and Karimi, Fariba and Gal\'{a}rraga, Luis and M\'{e}ndez, Gonzalo G. and Esp\'{i}n-Noboa, Lisette},
  title     = {Whose Name Comes Up? I: Auditing LLM-Based Scholar Recommendations},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.00074},
  note      = {v1.0.0, under-review}
}
```

## Contributors

| Contributor | Contributed to |
|-------------|----------------|
| [**DanieleBarolo**](https://github.com/DanieleBarolo) | LLMCaller, GTBuilder/OpenAlex_API_Pipeline |
| [**lisette-espin**](https://github.com/lisette-espin) | Auditor, GTBuilder/APS |


## License

See the repository for license information.
