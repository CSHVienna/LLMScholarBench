# LLMCaller Directory Structure and Temperature Experiments - Comprehensive Guide

## Overview
LLMCaller is a sophisticated system for running experiments with multiple LLM providers (OpenRouter and Gemini). It supports temperature variations, handles retries intelligently, and manages complex experiment orchestration across 20+ models.

---

## 1. TEMPERATURE EXPERIMENTS CONFIGURATION

### Temperature Configuration Locations

#### A. Main Script Entry Point: `run_temperature_experiments.sh`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/run_temperature_experiments.sh`

**Temperature Values Defined**:
```bash
TEMPERATURES=(0 0.25 0.5 0.75 1 1.5 2)  # Line 14
```

**Key Features**:
- 7 temperature levels from 0 (deterministic) to 2 (highly creative)
- Supports three execution modes:
  1. `parallel` - Run both providers simultaneously
  2. `single` - Run single provider
  3. Both can be combined with specific temperature values

**Usage Examples**:
```bash
./run_temperature_experiments.sh                    # All temps, both providers
./run_temperature_experiments.sh parallel 0.5       # Specific temp for both
./run_temperature_experiments.sh single openrouter  # All temps, one provider
./run_temperature_experiments.sh single gemini 1.0  # Single temp, single provider
```

#### B. Model Configuration: `config/llm_setup.json`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/config/llm_setup.json`

**Temperature Settings Per Model**:
Each model has:
- `"temperature": 0.5` (default temperature)
- `"max_temperature": 2` (maximum supported)

Example for llama-3.1-70b:
```json
{
  "llama-3.1-70b": {
    "model": "meta-llama/llama-3.1-70b-instruct",
    "temperature": 0.5,
    "max_temperature": 2,
    "provider": "openrouter",
    "max_attempts": 3
  }
}
```

**Note on Temperature Scaling**:
- Some models (e.g., Gemini) have `max_temperature: 1`
- When temperature_override > 1 and max_temperature = 1, value is scaled: `actual_temp = temperature_override / 2`
- See `experiments/runner_openrouter_async.py` lines 103-108

---

## 2. MAIN EXECUTION SCRIPTS

### A. Entry Point: `main.py`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/main.py`

**Command Line Arguments**:
```python
--all-models-async          # Required: Run experiments fully async
--output-dir               # Override output directory
--category [CATEGORY]      # Run single category (top_k, biased_top_k, epoch, field, twins, seniority)
--variable [VAR]           # Run single variable (requires --category)
--provider [PROVIDER]      # Filter models: 'openrouter' or 'gemini'
--temperature [FLOAT]      # Override temperature for all models (0.0-2.0)
```

**Execution Flow**:
1. Loads available models (filtered by provider if specified)
2. Routes to appropriate async executor:
   - Gemini → `run_gemini_concurrent()`
   - OpenRouter → `OpenRouterAsyncRunner().run_all_models()`
   - Both → Runs both in parallel with `asyncio.gather()`

**Key Code** (lines 45-77):
```python
if args.provider == 'gemini':
    asyncio.run(run_gemini_concurrent(models, args.output_dir, args.category, 
                                     args.variable, args.temperature))
elif args.provider == 'openrouter':
    runner = OpenRouterAsyncRunner(args.output_dir, temperature_override=args.temperature)
    asyncio.run(runner.run_all_models(models, args.category, args.variable))
else:
    # Run both in parallel
    asyncio.run(run_both_providers())
```

### B. Parallel Execution: `slurm_both_parallel.sh`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/slurm_both_parallel.sh`

**Execution Strategy**:
- Runs OpenRouter and Gemini **simultaneously** in background
- Waits for both to complete before exiting
- Output logged to `logs/openrouter_inline.log` and `logs/gemini_inline.log`

```bash
# Line 46-51: Both providers run in parallel
python3 main.py --all-models-smart --provider openrouter --batch-size 15 > logs/openrouter_inline.log 2>&1 &
OPENROUTER_PID=$!

python3 main.py --all-models-smart --provider gemini > logs/gemini_inline.log 2>&1 &
GEMINI_PID=$!

wait $OPENROUTER_PID
wait $GEMINI_PID
```

---

## 3. EXPERIMENT RUNNER IMPLEMENTATION

### A. OpenRouter Async Runner: `experiments/runner_openrouter_async.py`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/experiments/runner_openrouter_async.py`

**Class**: `OpenRouterAsyncRunner`

**Temperature Override Logic** (lines 103-109):
```python
if temperature_override is not None:
    max_temp = config.get('max_temperature', 2)
    if max_temp == 1:
        actual_temp = temperature_override / 2  # Scale down for models with max_temp=1
    else:
        actual_temp = temperature_override
    config['temperature'] = actual_temp
```

**Retry Strategy - Two-Level**:
1. **Infrastructure Retries** (5 attempts): For API failures
   - Handles: RateLimitError, BadRequestError, JSONDecodeError, APIConnectionError
   - Wait times: 5s, 10s, 20s, 40s between retries

2. **Validation Retries** (3 attempts): For response quality issues
   - Handles: Invalid format, schema validation failures
   - Only retries if validation fails

**Key Methods**:
- `run_single_experiment()`: Core retry logic (lines 92-192)
- `_get_or_create_run_dir()`: Directory management (lines 68-90)
- `run_all_models()`: Orchestrates parallel execution

### B. Gemini Concurrent Runner: `run_gemini_concurrent.py`
**File**: `/Users/barolo/LLMScholar-Audits/LLMCaller/run_gemini_concurrent.py`

**Temperature Handling** (lines 18-20):
```python
if temperature_override is not None:
    config['temperature'] = temperature_override
```
- No scaling applied (Gemini models handled differently)

**Async Function**: `run_single_gemini_experiment()`
- Handles both regular and grounded Gemini responses
- Full response saving before validation
- Same retry pattern as OpenRouter

---

## 4. MODEL CONFIGURATION

### Configured Models in `config/llm_setup.json`

**OpenRouter Models** (17 total):
1. llama-3.3-8b
2. llama-4-scout
3. llama-4-mav
4. gpt-oss-20b
5. gpt-oss-120b
6. qwen3-8b
7. qwen3-14b
8. qwen3-32b
9. qwen3-30b-a3b-2507
10. qwen3-235b-a22b-2507
11. gemma-3-12b-it
12. gemma-3-27b-it
13. mistral-small-3.2-24b
14. mistral-medium-3
15. llama-3.1-70b
16. llama-3.3-70b
17. llama-3.1-405b
18. grok-4-fast
19. deepseek-chat-v3.1
20. deepseek-r1-0528

**Gemini Models** (4 total):
1. gemini-2.5-flash
2. gemini-2.5-flash-grounded
3. gemini-2.5-pro
4. gemini-2.5-pro-grounded

**Configuration Structure Per Model**:
```json
{
  "model": "meta-llama/llama-3.1-70b-instruct",
  "max_attempts": 3,
  "temperature": 0.5,
  "max_temperature": 2,
  "stop": null,
  "stream": false,
  "system_message_ref": "physics_research_assistant",
  "provider": "openrouter",
  "sub_provider": "deepinfra/base",
  "quantization": "bf16",
  "latency": "0.47s",
  "throughput": "19.63tps",
  "total_context": "131.1K",
  "max_output": "131.1K",
  "host_country": "US",
  "input_price": "0.40",
  "output_price": "0.40"
}
```

---

## 5. API CLIENTS

### A. OpenRouter Async Client: `api/openrouter_api_async.py`

**Key Features**:
- AsyncOpenAI client configured for OpenRouter
- No rate limiting (fully async)
- Sub-provider enforcement support

**Temperature Applied In**:
- Direct pass-through from config to API call
- Line 84: `"temperature": self.config['temperature']`

### B. Gemini API Client: `api/gemini_api.py`

**Key Features**:
- Vertex AI support
- Grounded and non-grounded modes
- Concurrent execution support

---

## 6. OUTPUT DIRECTORY STRUCTURE

### Experiment Output Hierarchy
```
experiments/
├── config_[MODEL_NAME]/                     # One per model
│   ├── llm_setup.json                       # Model config copy
│   └── run_[TIMESTAMP]/                     # One per run (date+time)
│       ├── experiment_runner.log            # Detailed execution log
│       ├── experiment_summary.json           # Quick reference of all attempts
│       ├── [CATEGORY]_[VARIABLE]/           # One per experiment
│       │   ├── attempt1_[TIMESTAMP].json    # Full response + validation
│       │   ├── attempt2_[TIMESTAMP].json    # (if retry needed)
│       │   └── attempt3_[TIMESTAMP].json    # (if more retries)
│       ├── [CATEGORY2]_[VARIABLE2]/
│       │   └── attempt1_[TIMESTAMP].json
│       └── ...
└── global_runs/
    └── run_[TIMESTAMP].log                  # Global execution log (all models)
```

**Timestamp Format**: `YYYYMMDD_HHMMSS` (e.g., `20251007_234515`)

---

## 7. OUTPUT FILE STRUCTURES

### A. Individual Attempt JSON: `attemptN_[TIMESTAMP].json`
**Location**: `experiments/config_[MODEL]/run_[TIME]/[CATEGORY]_[VARIABLE]/`

**Structure** (from actual example):
```json
{
  "category": "twins",
  "variable": "famous_male",
  "prompt": "[Full prompt text...]",
  "attempt": 1,
  "infrastructure_retries": 5,
  "error": {
    "error_type": "APIConnectionError",
    "message": "Connection error."
  },
  "validation_result": {
    "is_valid": false,
    "message": "Infrastructure failure - API call failed",
    "extracted_data": null
  },
  "full_api_response": {
    "error_from_exception": "Connection error.",
    "exception_type": "APIConnectionError",
    "note": "API call itself failed after infrastructure retries"
  }
}
```

### B. Experiment Summary: `experiment_summary.json`
**Location**: `experiments/config_[MODEL]/run_[TIME]/`

**Structure**:
```json
{
  "twins": {
    "politic_female": {
      "attempts": [
        {
          "attempt": 1,
          "timestamp": "20251007_234614",
          "is_valid": true
        }
      ],
      "latest_attempt": 1
    },
    "famous_male": {
      "attempts": [
        {
          "attempt": 1,
          "timestamp": "20251007_234642",
          "is_valid": false,
          "error": {
            "error_type": "APIConnectionError",
            "message": "Connection error."
          }
        }
      ],
      "latest_attempt": 1
    }
  },
  "epoch": { ... },
  "field": { ... },
  "biased_top_k": { ... },
  "top_k": { ... },
  "seniority": { ... }
}
```

---

## 8. EXPERIMENT CATEGORIES AND VARIABLES

**Categories** (6 total):
1. `twins` - Statistical twin scientists
2. `epoch` - Different time periods
3. `field` - Different research fields
4. `biased_top_k` - Top-k with bias variations
5. `top_k` - Simple top-k lists
6. `seniority` - Career stage

**Variable Examples**:
- twins: famous_male, famous_female, fictitious_male, fictitious_female, movie_male, movie_female, politic_male, politic_female, random_male, random_female
- epoch: 1950s, 2000s
- field: PER, CM&MP
- biased_top_k: top_100_bias_gender_male, top_100_bias_gender_female, top_100_bias_gender_equal, top_100_bias_gender_neutral, top_100_bias_ethnicity_asian, top_100_bias_ethnicity_black, top_100_bias_ethnicity_white, top_100_bias_ethnicity_latino, top_100_bias_ethnicity_equal, top_100_bias_diverse, top_100_bias_citations_high, top_100_bias_citations_low
- top_k: top_5, top_100
- seniority: early_career, senior

---

## 9. STORAGE AND SAVING LOGIC

### A. `storage/saver.py` - Individual Result Saving
```python
def save_attempt(result, run_dir):
    # Creates: run_dir/[CATEGORY]_[VARIABLE]/attempt[N]_[TIMESTAMP].json
    # Always saves complete result with prompt, response, validation
```

### B. `storage/summarizer.py` - Summary Updating
```python
def update_summary(result, run_dir):
    # Updates: run_dir/experiment_summary.json
    # Tracks all attempts per variable with validation status
```

---

## 10. LOGGING IMPLEMENTATION

### Log Levels and Locations

**Model-Specific Logs**:
- Location: `experiments/config_[MODEL]/run_[TIME]/experiment_runner.log`
- Content: All API calls, retries, validations per model

**Global Logs**:
- Location: `experiments/global_runs/run_[TIMESTAMP].log`
- Content: High-level experiment coordination

**Infrastructure Error Tracking** (from logs):
```
🔄 API infrastructure error (APIConnectionError): llama-3.1-70b:twins:fictitious_female - Infra retry 1/5 after 5s
🔄 API infrastructure error (APIConnectionError): llama-3.1-70b:twins:fictitious_female - Infra retry 2/5 after 10s
```

---

## 11. TEMPERATURE EXPERIMENT WORKFLOW

### Complete Flow for Temperature Variation

1. **Script Initiation**
   ```bash
   ./run_temperature_experiments.sh parallel 0.5
   ```

2. **Main.py Processing**
   - Parses `--temperature 0.5` argument
   - Loads all models for both providers
   - Routes to OpenRouter and Gemini runners

3. **Temperature Override in Model Config**
   - OpenRouter runner checks: `temperature_override = 0.5`
   - Applies scaling if max_temperature = 1
   - Sets `config['temperature'] = actual_temp`

4. **API Calls**
   - All models use overridden temperature value
   - API receives: `"temperature": 0.5` in request

5. **Response Collection**
   - Responses saved with temperature info in directory structure
   - Summary tracks all attempts with timestamps

6. **Output Organization**
   - Results at: `experiments/config_[MODEL]/run_[TIME]/[CATEGORY]_[VAR]/attempt1_[TIME].json`
   - Contains: Full prompt, response, validation status, error tracking

---

## 12. RETRY MECHANISM FOR TEMPERATURE EXPERIMENTS

### Infrastructure Failure Handling
```python
max_infra_retries = 5
wait_times = [5, 10, 20, 40]  # seconds between retries

# If API fails (rate limit, connection, JSON error):
for infra_retry in range(max_infra_retries):
    try:
        api_response = await api_client.generate_response(prompt)
        break  # Success!
    except Exception:
        wait(wait_times[min(infra_retry, len(wait_times)-1)])
        continue
```

### Validation Failure Handling
```python
max_validation_attempts = 3

for validation_attempt in range(1, max_validation_attempts + 1):
    # Infrastructure retries happen here
    api_response = await get_response_with_infra_retries()
    
    if validate(api_response):
        return success
    elif validation_attempt < max_validation_attempts:
        # Retry entire request (with new infrastructure retries)
        continue
```

---

## 13. KEY CONFIGURATION OVERRIDES

### Environment Variables
```bash
LLMCALLER_CREDENTIALS    # Path to credentials folder
LLMCALLER_OUTPUT         # Path to experiments output folder
```

### Command Line Overrides
```bash
--temperature 0.75       # Override all models' temperature
--provider openrouter    # Run only OpenRouter (skip Gemini)
--category twins         # Run only "twins" experiments
--variable famous_male   # Run only specific variable
```

---

## 14. CREDENTIAL MANAGEMENT

**Required Folder Structure**:
```
credentials/
├── .env                           # OPENROUTER_API_KEY
└── .keys/
    ├── config.ini                 # Google Cloud project config
    └── service-account.json       # GCP service account JSON
```

**Loaded By**:
- `api/openrouter_api_async.py`: Lines 20-43 (loads from .env)
- `api/gemini_api.py`: Loads from .keys/ directory

---

## 15. EXAMPLE: RUNNING TEMPERATURE EXPERIMENTS

### Scenario 1: Single Temperature for All Models
```bash
cd /Users/barolo/LLMScholar-Audits/LLMCaller
source .venv/bin/activate
python3 main.py --all-models-async --temperature 1.5 --output-dir /path/to/output
```

**Result**:
- All 20+ models use temperature = 1.5
- Results saved in separate model folders with timestamp
- experiment_summary.json shows all attempts

### Scenario 2: All Temperatures for Both Providers
```bash
./run_temperature_experiments.sh
```

**Result**:
- 7 temperature values (0, 0.25, 0.5, 0.75, 1, 1.5, 2)
- Both OpenRouter and Gemini run in parallel
- OpenRouter and Gemini output separate result directories
- 7 separate runs total

### Scenario 3: Specific Temperature for OpenRouter Only
```bash
python3 main.py --all-models-async --provider openrouter --temperature 0.5
```

**Result**:
- Only OpenRouter models tested
- All use temperature = 0.5
- Results in experiments/config_[MODEL]/run_[TIME]/

---

## 16. SUMMARY OF KEY FILES

| File | Purpose | Key Info |
|------|---------|----------|
| `run_temperature_experiments.sh` | Temperature experiment orchestration | Defines 7 temp values, runs parallel |
| `main.py` | Entry point | Routes to OpenRouter or Gemini runner |
| `experiments/runner_openrouter_async.py` | OpenRouter execution | 5-level infra retry, 3-level validation retry |
| `run_gemini_concurrent.py` | Gemini execution | Async concurrent execution |
| `config/llm_setup.json` | Model configurations | Temperature defaults, max_temperature per model |
| `api/openrouter_api_async.py` | OpenRouter API client | Pure async, no rate limiting |
| `slurm_both_parallel.sh` | Parallel run script | Runs both providers simultaneously |
| `storage/saver.py` | Result persistence | Saves individual attempt JSONs |
| `storage/summarizer.py` | Summary tracking | Updates experiment_summary.json |

---

## 17. CRITICAL TEMPERATURE VARIATION POINTS

1. **Default Temperature**: `config/llm_setup.json` - `"temperature": 0.5`
2. **Max Temperature**: `config/llm_setup.json` - `"max_temperature": 2`
3. **Override Logic**: `experiments/runner_openrouter_async.py` - lines 103-109
4. **API Pass-Through**: `api/openrouter_api_async.py` - line 84
5. **Gemini Override**: `run_gemini_concurrent.py` - lines 18-20
6. **Script Temperatures**: `run_temperature_experiments.sh` - line 14

---

End of LLMCaller Structure and Temperature Experiments Guide
