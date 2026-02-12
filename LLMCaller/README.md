# LLMCaller

An automated framework for collecting data with multiple Large Language Models on specialized tasks related to identifying physicists who have published in APS journals across various filtering criteria (gender, ethnicity, time periods, research fields, etc.). It tests 14+ models across different providers.

## Structure

```
LLMCaller/
├── main.py                      # Entry point
├── api/                         # OpenRouter & Gemini async clients
├── config/                      # Model configs, experiment definitions, schemas
│   ├── llm_setup.json           # 24 model definitions
│   ├── category_variables.json  # 6 categories, 40+ variables
│   └── paths.json               # Credentials & output paths
├── prompts/                     # Dynamic prompt generation
├── validation/                  # JSON schema validation
├── storage/                     # Result persistence & summaries
└── experiments/                 # Async runners for OpenRouter/Gemini
```

## Setup

```bash
# 1. Create config/paths.json
{
  "credentials_dir": "/path/to/credentials",
  "output_dir": "/path/to/experiments"
}

# 2. Populate credentials/
#    - .env (OpenRouter API key)
#    - .keys/config.ini (GCP settings)
#    - .keys/service-account.json (Google credentials)

# 4. Install dependencies (with uv or conda)
uv pip install -r requirements.txt
```

## Usage

```bash
# Run all models asynchronously
python3 main.py --all-models-async

# Run specific provider
python3 main.py --all-models-async --provider openrouter
python3 main.py --all-models-async --provider gemini

# Filter by category/variable
python3 main.py --all-models-async --category top_k --variable top_5

# Override temperature
python3 main.py --all-models-async --temperature 1.5
```

## Experiment Categories

- **top_k**: Top 5 or 100 physicists
- **biased_top_k**: Bias testing (gender, ethnicity, citations)
- **field**: Subfield-specific (PER, CM&MP)
- **epoch**: Time-restricted (1950s, 2000s)
- **seniority**: Early career vs senior
- **twins**: Real vs fictitious scientists

### Notes

In certain experiment prompts, fictitious names may be required for variables or specific task requirements. For example, the names "Agandaur Heilamin" (Male) and "Huethea Arabalar" (Female) were generated for use in the LLMCaller experiments on **19/09/2024 at 16:00**. These names were created using [Random Word Generator](https://randomwordgenerator.com/name.php), selecting **Fantasy Names** with no specific regional origin and once each for male and female genders.


## Output

Results stored at: `experiments/config_[model]/run_[timestamp]/[category]_[variable]/attempt1_[timestamp].json`

Each result contains:
- Generated prompt
- Full API response
- Validation results
- Extracted data
- Metadata (prompt, temp value etc.)

## Key Features

- Async parallel execution across 14+ models
- Multi-provider support (OpenRouter, Gemini)
- JSON schema validation per category with retry (up to 3 attempts)
- Logging and traceability
- Structured result storage

