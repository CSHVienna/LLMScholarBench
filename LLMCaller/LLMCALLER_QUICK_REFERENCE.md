# LLMCaller - Quick Reference Guide

## File Paths (Absolute Paths)

```
Project Root:  /Users/barolo/LLMScholar-Audits/LLMCaller/

Core Files:
- run_temperature_experiments.sh    (orchestrates temp experiments)
- main.py                           (entry point)
- slurm_both_parallel.sh           (runs both providers in parallel)

Configuration:
- config/llm_setup.json            (all model configs + temperature settings)
- config/loader.py                 (config loading utilities)

Runners:
- experiments/runner_openrouter_async.py  (OpenRouter execution engine)
- run_gemini_concurrent.py                (Gemini execution engine)

API Clients:
- api/openrouter_api_async.py      (OpenRouter API client)
- api/gemini_api.py                (Gemini/Vertex AI client)

Storage:
- storage/saver.py                 (saves individual attempts)
- storage/summarizer.py            (updates experiment_summary.json)

Logging:
- logs/                            (execution logs directory)
- experiments/config_*/run_*/experiment_runner.log  (per-model logs)
- experiments/global_runs/run_*.log (global execution logs)

Output:
- experiments/                     (root output directory)
  - config_[model_name]/           (per model)
    - run_[YYYYMMDD_HHMMSS]/       (per run)
      - [category]_[variable]/     (per experiment)
        - attempt[N]_[timestamp].json
      - experiment_summary.json
      - experiment_runner.log
```

---

## Quick Start Commands

### Run with Specific Temperature
```bash
cd /Users/barolo/LLMScholar-Audits/LLMCaller
python3 main.py --all-models-async --temperature 1.0
```

### Run Temperature Script (All 7 Temperatures)
```bash
./run_temperature_experiments.sh              # All temps, both providers
./run_temperature_experiments.sh parallel 0.5 # Specific temp
./run_temperature_experiments.sh single openrouter  # One provider
```

### Run Both Providers Simultaneously
```bash
./slurm_both_parallel.sh
```

### Run Single Provider
```bash
python3 main.py --all-models-async --provider openrouter
python3 main.py --all-models-async --provider gemini
```

### Run Single Category or Variable
```bash
python3 main.py --all-models-async --category twins
python3 main.py --all-models-async --category twins --variable famous_male
```

---

## Temperature Values

### Predefined in Script
```bash
# From run_temperature_experiments.sh line 14
TEMPERATURES=(0 0.25 0.5 0.75 1 1.5 2)

# 0    = Deterministic (always same response)
# 0.25 = Very structured
# 0.5  = Default (balanced)
# 0.75 = More creative
# 1.0  = Creative
# 1.5  = Very creative
# 2.0  = Extremely creative (max for most models)
```

### Default per Model
```json
{
  "temperature": 0.5,        // Default used if not overridden
  "max_temperature": 2       // Maximum the model accepts
}
```

---

## Models Available

### OpenRouter (20 Models)

**Llama Family**:
- llama-3.3-8b
- llama-3.3-70b
- llama-3.1-70b
- llama-3.1-405b
- llama-4-scout
- llama-4-mav

**Qwen Family**:
- qwen3-8b
- qwen3-14b
- qwen3-32b
- qwen3-30b-a3b-2507
- qwen3-235b-a22b-2507

**Other Models**:
- gpt-oss-20b
- gpt-oss-120b
- gemma-3-12b-it
- gemma-3-27b-it
- mistral-small-3.2-24b
- mistral-medium-3
- grok-4-fast
- deepseek-chat-v3.1
- deepseek-r1-0528

### Gemini (4 Models)
- gemini-2.5-flash
- gemini-2.5-flash-grounded
- gemini-2.5-pro
- gemini-2.5-pro-grounded

---

## Experiment Categories & Variables

### Categories (6 Total)
```
1. twins         - Statistical twin scientists (10 variables)
2. epoch         - Time periods (2 variables)
3. field         - Research fields (2 variables)
4. biased_top_k  - Top-k with bias (12 variables)
5. top_k         - Simple top-k (2 variables)
6. seniority     - Career stage (2 variables)
```

### Total Combinations
- Per model: 6 categories × ~5 avg variables = ~30 experiments
- With retries: Up to 30 × 3 attempts = 90 max calls per model
- With all models: 24 models × 90 = ~2160 total possible calls

---

## Output Directory Structure

### Per-Run Output Tree
```
experiments/
└── config_llama-3.1-70b/
    ├── llm_setup.json
    └── run_20251007_234515/
        ├── experiment_runner.log        (detailed per-model log)
        ├── experiment_summary.json       (quick reference)
        ├── twins_famous_male/
        │   ├── attempt1_20251007_234642.json
        │   └── attempt2_20251007_234700.json  (if needed)
        ├── twins_famous_female/
        │   └── attempt1_20251007_234640.json
        ├── epoch_1950s/
        │   └── attempt1_20251007_234623.json
        └── ... (other category_variable folders)
```

### Timestamp Format
`YYYYMMDD_HHMMSS` Example: `20251007_234515` = Oct 7, 2025 @ 23:45:15

---

## Response Structure (attemptN_TIMESTAMP.json)

```json
{
  "category": "twins",
  "variable": "famous_male",
  "prompt": "... full prompt text ...",
  "attempt": 1,
  "infrastructure_retries": 0,           // Number of infra retries used
  "full_api_response": {
    "model": "meta-llama/llama-3.1-70b-instruct",
    "response": "... response text ...",
    "error_from_exception": null
  },
  "validation_result": {
    "is_valid": true,
    "message": "Valid JSON array",
    "extracted_data": [                  // Only if valid
      {"Name": "Scientist 1"},
      {"Name": "Scientist 2"}
    ]
  }
}
```

---

## Experiment Summary Structure (experiment_summary.json)

```json
{
  "twins": {
    "famous_male": {
      "attempts": [
        {
          "attempt": 1,
          "timestamp": "20251007_234642",
          "is_valid": true
        },
        {
          "attempt": 2,
          "timestamp": "20251007_234700",
          "is_valid": true
        }
      ],
      "latest_attempt": 2
    },
    "famous_female": {
      "attempts": [
        {
          "attempt": 1,
          "timestamp": "20251007_234640",
          "is_valid": false,
          "error": {
            "error_type": "APIConnectionError",
            "message": "Connection error."
          }
        }
      ],
      "latest_attempt": 1
    }
  }
}
```

---

## Retry Logic

### Two-Level Retry Strategy

#### Level 1: Infrastructure Retries (for API failures)
```
Max attempts: 5
Errors caught: APIConnectionError, RateLimitError, JSONDecodeError, BadRequestError
Wait times: 5s → 10s → 20s → 40s (exponential backoff)
```

#### Level 2: Validation Retries (for response quality)
```
Max attempts: 3 (from config: max_attempts)
Triggers on: Invalid JSON, schema validation failure
Each retry: Re-runs API call with new infrastructure retries
```

### Retry Flow
```
Validation Attempt 1
  ├─ Infrastructure Retry 1 → Fail? Wait 5s, Retry
  ├─ Infrastructure Retry 2 → Fail? Wait 10s, Retry
  ├─ Infrastructure Retry 3 → Fail? Wait 20s, Retry
  ├─ Infrastructure Retry 4 → Fail? Wait 40s, Retry
  └─ Infrastructure Retry 5 → Success!
       ├─ Validate response
       ├─ Valid? → Return Success
       └─ Invalid? → Try Validation Attempt 2

Validation Attempt 2 → (same infra retry sequence)
Validation Attempt 3 → (same infra retry sequence)
  └─ If still invalid → Return Last Attempt
```

---

## Temperature Override Points

### How Temperature Gets Applied

1. **Config Default** (llm_setup.json)
   - `"temperature": 0.5`
   - `"max_temperature": 2`

2. **CLI Override** (--temperature argument)
   ```bash
   python3 main.py --all-models-async --temperature 1.5
   ```

3. **Runner Processing** (runner_openrouter_async.py lines 103-109)
   ```python
   if temperature_override is not None:
       max_temp = config.get('max_temperature', 2)
       if max_temp == 1:
           actual_temp = temperature_override / 2
       else:
           actual_temp = temperature_override
       config['temperature'] = actual_temp
   ```

4. **API Call** (openrouter_api_async.py line 84)
   ```python
   "temperature": self.config['temperature']
   ```

### Scaling Rules
- OpenRouter models: No scaling (accept up to 2.0)
- Gemini models: Scale if needed (max 1.0)

---

## Environment Variables

```bash
# Required for credentials
export LLMCALLER_CREDENTIALS="/Users/barolo/Desktop/credentials"

# Optional for output
export LLMCALLER_OUTPUT="/Users/barolo/LLMScholar-Audits/LLMCaller/experiments"

# Check if set
echo $LLMCALLER_CREDENTIALS
echo $LLMCALLER_OUTPUT
```

---

## Log Locations

### Model-Specific Log
```
experiments/config_[MODEL_NAME]/run_[TIMESTAMP]/experiment_runner.log
```

**Example**:
```
experiments/config_llama-3.1-70b/run_20251007_234515/experiment_runner.log
```

**Content**: Per-model execution details
```
2025-10-07 23:45:22 - experiment_runner_llama-3.1-70b - INFO - 🚀 Running llama-3.1-70b
2025-10-07 23:45:23 - experiment_runner_llama-3.1-70b - INFO - ✅ llama-3.1-70b:twins:famous_male - Success
2025-10-07 23:45:24 - experiment_runner_llama-3.1-70b - WARNING - 🔄 API infrastructure error (APIConnectionError): llama-3.1-70b:twins:famous_female - Infra retry 1/5 after 5s
```

### Global Log
```
experiments/global_runs/run_[TIMESTAMP].log
```

**Content**: High-level coordination across all models

---

## Execution Times

### Typical Run Times (All Models)

```
OpenRouter (20 models) 
  × 30 experiments/model average
  × 1 retry attempt average
  = ~60-90 minutes (rate limited to ~15-20 calls/min)

Gemini (4 models)
  × 30 experiments/model
  × 1 retry attempt average
  = ~10-20 minutes (concurrent execution)

Both in Parallel = ~60-90 minutes (limited by OpenRouter)
```

---

## Common Issues & Solutions

### Issue: Temperature not applied
**Solution**: Check --temperature argument is passed correctly
```bash
python3 main.py --all-models-async --temperature 0.75  # Correct
python3 main.py --all-models-async 0.75                # Wrong
```

### Issue: Experiment fails immediately
**Check**:
1. Credentials exist: `ls $LLMCALLER_CREDENTIALS/.env`
2. Output directory writable: `mkdir -p $LLMCALLER_OUTPUT`
3. Models loaded: Check main.py output for model count

### Issue: Some experiments missing from output
**Reasons**:
- Still running (check if process active)
- Failed validation - check experiment_summary.json for error
- Infrastructure errors - see experiment_runner.log for details

---

## File Sizes & Storage

### Typical Output Per Model Run
```
experiment_runner.log       ~20-50 KB
experiment_summary.json     ~5-10 KB
Per attempt JSON            ~1-5 KB
Total per model run         ~100-500 KB

All 24 models (1 run each)   ~2.4-12 MB
7 temperature values         ~16-84 MB (7 runs each)
```

---

## Provider Filtering

### Run Only Specific Provider

```bash
# OpenRouter only (20 models)
python3 main.py --all-models-async --provider openrouter

# Gemini only (4 models)
python3 main.py --all-models-async --provider gemini

# Both (24 models, sequential)
python3 main.py --all-models-async
```

---

## Configuration Files Reference

### Main Config: config/llm_setup.json
- **Purpose**: All model definitions
- **Structure**: Flat list under "models" key
- **Keys per model**: 
  - model (API model ID)
  - provider (openrouter/gemini)
  - temperature, max_temperature
  - max_attempts
  - system_message_ref
  - Other metadata (latency, throughput, pricing)

### Categories Config: config/category_variables.json
- **Purpose**: Defines all experiment types and variables
- **Used by**: prompt generator
- **Updates**: When adding new experiment types

### System Messages: config/system_messages.json
- **Purpose**: System prompts for different roles
- **Used by**: Each model config via system_message_ref
- **Default**: physics_research_assistant

---

## Integration Points

### If Adding New Models
1. Edit `config/llm_setup.json` - add model config
2. Edit `config/loader.py` - if special handling needed
3. Test: `python3 main.py --all-models-async`

### If Adding New Experiment Category
1. Edit `config/category_variables.json` - add category + variables
2. Edit `config/schemas/[category].json` - define schema
3. Edit `prompts/generator.py` - add prompt generation
4. Test: `python3 main.py --all-models-async --category [new]`

### If Changing Temperature Scaling
1. Edit `experiments/runner_openrouter_async.py` - lines 103-109
2. Edit `run_gemini_concurrent.py` - lines 18-20 (if needed)
3. Test: `python3 main.py --all-models-async --temperature [value]`

---

## Key Code Sections

| Section | File | Lines | Purpose |
|---------|------|-------|---------|
| Temperature parsing | main.py | 28-29 | Read --temperature arg |
| OpenRouter override | runner_openrouter_async.py | 103-109 | Apply temp override |
| Gemini override | run_gemini_concurrent.py | 18-20 | Apply temp to Gemini |
| API call with temp | openrouter_api_async.py | 84 | Send to API |
| Result saving | storage/saver.py | 5-20 | Save attempt JSON |
| Summary updating | storage/summarizer.py | 5-37 | Update summary.json |

---

## Testing Temperatures

### Quick Test (One Model, One Experiment)
```bash
python3 main.py --all-models-async --provider openrouter \
  --temperature 1.0 --category twins --variable famous_male
```

**Output**: 
- Check: `experiments/config_llama-3.3-8b/run_[TIME]/twins_famous_male/`
- Look for: Temperature value in API call logs

---

End of Quick Reference Guide
