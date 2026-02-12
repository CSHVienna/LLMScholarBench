# LLMScholarBench

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

Code to **query LLMs** and collect scholar-recommendation responses. It supports multiple providers (e.g. OpenRouter, Gemini), configurable prompts and schemas, and structured storage of outputs for downstream auditing.

- **Location**: `LLMCaller/`
- **Features**: Async execution across many models, JSON schema validation, experiment categories (top-k experts, field/epoch/seniority, bias tests, twins), and configurable temperatures.
- **Setup**: See `LLMCaller/README.md` for credentials, `config/paths.json`, and installation (e.g. `pip install -r LLMCaller/requirements.txt`).

### 2. Auditor

Code to **analyze LLM outputs** against ground truth. It runs preprocessing pipelines (valid answers, factuality, disciplines, similarities) and provides notebooks for consistency, factuality, demographics, prestige, and similarity analyses.

- **Location**: `Auditor/`
- **Features**: Batch preprocessing scripts, factuality checks (author, field, epoch, seniority), demographic and similarity metrics, and visualization notebooks.
- **Setup**: Set `PYTHONPATH` from `Auditor/code/` and use the batch scripts; see `Auditor/README.md` for the full pipeline.

### 3. GTBuilder

Module to **standardize ground-truth data** so analyses can be run with any compatible GT source. It defines pipelines and schemas to build comparable scholar benchmarks. Currently implemented for **American Physical Society (APS)** publications (physics domain), with OpenAlex used for author/institution metadata.

- **Location**: `GTBuilder/`
- **Contents**:
  - **APS**: Scripts and notebooks to build author stats, demographics, affiliations, coauthorship networks, disciplines, and publication classes from APS + OpenAlex.
  - **OpenAlex API Pipeline**: Scripts to fetch publications, authors, and institutions from the OpenAlex API (used by the APS pipeline).

## Installation

1. Clone the repository and use Python 3.11.11:

   ```bash
   git clone https://github.com/<org>/LLMScholarBench.git
   cd LLMScholarBench
   # Use Python 3.11.11 (e.g. pyenv local 3.11.11)
   ```

2. Install dependencies per module (each has its own `requirements.txt` if needed):

   ```bash
   pip install -r LLMCaller/requirements.txt
   # Install Auditor/GTBuilder deps as required by their scripts/notebooks
   ```

3. Configure and run each module according to its README (`LLMCaller/`, `Auditor/`, `GTBuilder/`).



## Related papers

| Version | Description | Link |
|--------|-------------|------|
| **v2.0** | Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation | [arXiv:2602.08873](https://arxiv.org/abs/2602.08873) *(under review)* |
| **v1.0** | Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations | [arXiv:2506.00074](https://arxiv.org/abs/2506.00074) *(under review)* |



## Citation

If you use LLMScholarBench in your work, please cite the benchmark and/or the relevant paper(s) below.

**BibTeX (v2.0):**

```bibtex
@article{espinnoboa2026llmscholarbench,
  author    = {Espín-Noboa, Lisette and Méndez, Gonzalo G.},
  title     = {Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation},
  journal   = {arXiv preprint},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.08873},
  note      = {v2.0, under review}
}
```

**BibTeX (v1.0):**

```bibtex
@article{barolo2025llmscholaraudits,
  author    = {Barolo, Daniele and Valentin, Chiara and Karimi, Fariba and Galárraga, Luis and Méndez, Gonzalo G. and Espín-Noboa, Lisette},
  title     = {Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.00074},
  note      = {v1.0, under review}
}
```

## Contributors

- [**lisette-espin**](https://github.com/lisette-espin)
- [**DanieleBarolo**](https://github.com/DanieleBarolo)


## License

See the repository for license information.
