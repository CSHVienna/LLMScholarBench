# Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations

This repository provides a framework for auditing and evaluating LLM-based scholar recommendations, along with preprocessing the APS dataset. It contains the following components:

## Folders

### 1. `APS`
This folder contains the code to preprocess the APS dataset. It includes scripts for cleaning, transforming, and preparing the data for further analysis or model training.

### 2. `Auditor`
This folder includes the code for running audits, which are used to evaluate LLM responses. The audit process involves assessing the quality, correctness, and bias of LLM outputs based on predefined criteria.

### 3. `Evaluator` *(Deprecated for Audits)*
This folder contains legacy code for running audits (evaluations) of LLM responses. While the audit functionality is now deprecated, the code for preprocessing (cleaning) LLM responses is still functional and useful for data preparation.

### 4. `LLMCaller`
This folder contains the code for collecting data from large language models (LLMs). It includes scripts for making API calls to LLMs and retrieving their responses for evaluation.
