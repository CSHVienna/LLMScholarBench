# LLMCaller

## Overview

LLMCaller is a modular and flexible system designed to run experiments with various Large Language Models (LLMs) on specific tasks related to identifying and compiling lists of physicists who have published in American Physical Society (APS) journals. This system allows for easy configuration of different models, systematic execution of experiments, and organized storage of results.

## Key Features

- Support for 14+ models across multiple API providers (OpenRouter, Gemini)
- Smart Queue System for optimal API efficiency and zero wasted calls
- Parallel execution support (run multiple providers simultaneously)
- Modular architecture for easy extension and maintenance
- Randomized experiment execution to minimize ordering bias
- Centralized credential management
- Robust error handling and comprehensive logging
- Structured storage of experiment results and configurations

## System Architecture

The LLMCaller system is composed of several modules, each with a specific responsibility:

1. **Main Script** (`main.py`): Entry point for running experiments.
2. **Experiment Runner** (`experiments/runner.py`): Orchestrates the execution of experiments.
3. **API Clients** (`api/`): Multi-provider support via factory pattern
   - `api_factory.py`: Creates appropriate client based on provider
   - `openrouter_api.py`: OpenRouter API with rate limiting
   - `gemini_api.py`: Google Gemini/Vertex AI
   - `openai_api.py`: OpenAI-compatible API client
4. **Configuration Management** (`config/`): Loads and validates experiment configurations.
5. **Prompt Generation** (`prompts/generator.py`): Creates prompts for each experiment variable.
6. **Result Storage** (`storage/`): Saves experiment results and maintains summaries.
7. **Logging** (`logs/setup.py`): Sets up logging for the experiment runs.
8. **Validation** (`validation/validator.py`): Validates LLM responses.
9. **Smart Queue System** (`utils/smart_queue.py`): Advanced batching with cross-model optimization.

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

## API Providers & Models

### OpenRouter (10 models)
Free-tier models with rate limiting (15-20 calls/min):
- LLaMA: `llama-3.3-8b`, `llama-3.3-70b`, `llama-3.1-405b`
- Qwen: `qwen3-8b`, `qwen3-235b`
- Others: `gpt-oss-20b`, `mistral-small-3.2`, `gemma-3-27b`, `deepseek-chat-v3.1`, `deepseek-r1`

### Gemini (4 models)
Google Vertex AI with concurrent execution:
- `gemini-2.5-flash`, `gemini-2.5-flash-grounded`
- `gemini-2.5-pro`, `gemini-2.5-pro-grounded`

## Usage

### Recommended: Run all models with smart queue
```bash
# OpenRouter (60-90 min)
python main.py --all-models-smart --provider openrouter --batch-size 20

# Gemini (10-20 min)
python main.py --all-models-smart --provider gemini
```

### Single model
```bash
python main.py --model llama-3.3-8b
```

### Parallel execution (both providers)
```bash
./slurm_both_parallel.sh
```

See [SMART_QUEUE.md](SMART_QUEUE.md) for detailed documentation.

## Credentials Setup

### Required files in credentials directory:

```bash
credentials/
├── .env                    # OpenRouter API key
└── .keys/                  # Google Cloud config directory
    ├── config.ini         # GCP project settings
    └── service-account.json
```

### Setup

```bash
# Set environment variable
export LLMCALLER_CREDENTIALS="/path/to/credentials"
export LLMCALLER_OUTPUT="/path/to/experiments"

# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > $LLMCALLER_CREDENTIALS/.env

# Create .keys/config.ini
mkdir -p $LLMCALLER_CREDENTIALS/.keys
# Add Google Cloud settings and service account JSON
```

## Directory Structure

After running experiments:

```
experiments/
└── config_[model_name]/
    ├── llm_setup.json
    ├── run_[timestamp1]/
    │   ├── experiment_runner.log
    │   ├── experiment_summary.json
    │   ├── [category1]_[variable1]/
    │   │   ├── attempt1_[timestamp].json  # Full result: prompt, response, validation
    │   │   └── attempt2_[timestamp].json  # Retry if needed
    │   └── [category2]_[variable2]/
    │       └── attempt1_[timestamp].json
    └── run_[timestamp2]/
        └── ...

logs/
├── openrouter_inline.log
└── gemini_inline.log
```

## Extending the System

### Add new models
Edit `config/llm_setup.json`:
```json
{
  "your-model": {
    "model": "provider/model-id",
    "provider": "openrouter",
    "temperature": 0,
    "max_attempts": 3
  }
}
```

### Add new experiment categories
1. Update `config/category_variables.json`
2. Create schema in `config/schemas/[category].json`
3. Update `prompts/generator.py`

## Running on a Server (Scheduled Execution)

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd LLMScholar-Audits/LLMCaller
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure credentials
export LLMCALLER_CREDENTIALS="/path/to/credentials"
export LLMCALLER_OUTPUT="/path/to/experiments"
# Setup .env and .keys/ as described in Credentials Setup

# 3. Edit paths in slurm_both_parallel.sh
# Update LLMCALLER_CREDENTIALS and LLMCALLER_OUTPUT

# 4. Test run
chmod +x slurm_both_parallel.sh
./slurm_both_parallel.sh

# 5. Schedule with cron (3x daily: midnight, 8am, 4pm)
crontab -e
# Add: 0 0,8,16 * * * cd /full/path/to/LLMCaller && ./slurm_both_parallel.sh
```

### Performance
- **Parallel (both providers)**: ~60-90 min (recommended)
- **OpenRouter only**: ~60-90 min (rate limited)
- **Gemini only**: ~10-20 min (concurrent)

Logs: `logs/` directory with timestamps.

For more detailed information, see [SMART_QUEUE.md](SMART_QUEUE.md) and inline documentation in the respective Python files.

## Fictitious Twin Names

In certain experiment prompts, fictitious names may be required for variables or specific task requirements. For example, the names "Agandaur Heilamin" (Male) and "Huethea Arabalar" (Female) were generated for use in the LLMCaller experiments on **19/09/2024 at 16:00**. These names were created using [Random Word Generator](https://randomwordgenerator.com/name.php), selecting **Fantasy Names** with no specific regional origin and once each for male and female genders.
