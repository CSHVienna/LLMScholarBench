# LLMCaller

## Overview

LLMCaller is a modular and flexible system designed to run experiments with various Large Language Models (LLMs) on specific tasks related to identifying and compiling lists of physicists who have published in American Physical Society (APS) journals. This system allows for easy configuration of different models, systematic execution of experiments, and organized storage of results.

## Key Features

- Support for multiple LLM configurations (e.g., Gemma 2 9b, LLaMA 3 (8b, 70b), Mixtral 8x7b)
- Modular architecture for easy extension and maintenance
- Randomized experiment execution to minimize ordering bias
- Robust error handling and logging
- Structured storage of experiment results and configurations

## System Architecture

The LLMCaller system is composed of several modules, each with a specific responsibility:

1. **Main Script** (`main.py`): Entry point for running experiments.
2. **Experiment Runner** (`experiments/runner.py`): Orchestrates the execution of experiments.
3. **API Client** (`api/groq_api.py`): Handles communication with the LLM API.
4. **Configuration Management** (`config/`): Loads and validates experiment configurations.
5. **Prompt Generation** (`prompts/generator.py`): Creates prompts for each experiment variable.
6. **Result Storage** (`storage/`): Saves experiment results and maintains summaries.
7. **Logging** (`logs/setup.py`): Sets up logging for the experiment runs.
8. **Validation** (`validation/validator.py`): Validates LLM responses.

## How It Works

1. The user selects a model configuration and runs the main script.
2. The system loads the appropriate configuration and sets up the experiment environment.
3. Experiments are run for each category-variable pair in a randomized order.
4. For each experiment:
   - A prompt is generated based on the category and variable.
   - The prompt is sent to the LLM via the API client.
   - The response is validated and stored.
   - If the response is invalid, the system retries up to a configured maximum number of attempts.
5. Results are saved in a structured directory format, including logs and summaries.

## Usage

To run an experiment with a specific model:

```bash
python main.py --model MODEL_NAME
```

Replace `MODEL_NAME` with one of the following options:
- `gemma2-9b`
- `llama3-70b`
- `llama3-8b`
- `mixtral-8x7b`

## Directory Structure

After running experiments, your directory structure will look like this:

```
experiments/
└── config_[model_name]/
    ├── llm_setup.json
    ├── run_[timestamp1]/
    │   ├── experiment_runner.log
    │   ├── experiment_summary.json
    │   ├── [category1]_[variable1]/
    │   │   ├── attempt1_[timestamp].json
    │   │   └── attempt2_[timestamp].json
    │   └── [category2]_[variable2]/
    │       └── attempt1_[timestamp].json
    └── run_[timestamp2]/
        ├── experiment_runner.log
        ├── experiment_summary.json
        └── ...
```

## Extending the System

To add new models or modify existing ones, update the `config/model_configs.json` file.

To add new experiment categories or variables, modify the `config/category_variables.json` file and update the prompt generation logic in `prompts/generator.py`.

## Troubleshooting

If you encounter issues:
1. Check the experiment logs in the run directory.
2. Ensure your API key is correctly set in the environment variables.
3. Verify that the selected model is available and correctly configured.

For more detailed information on each module, refer to the inline documentation in the respective Python files.

## Fictitious Twin Names

In certain experiment prompts, fictitious names may be required for variables or specific task requirements. For example, the names "Agandaur Heilamin" (Male) and "Huethea Arabalar" (Female) were generated for use in the LLMCaller experiments on **19/09/2024 at 16:00**. These names were created using [Random Word Generator](https://randomwordgenerator.com/name.php), selecting **Fantasy Names** with no specific regional origin and once each for male and female genders.
