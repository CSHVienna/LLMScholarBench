# Running LLM Experiments with Provider Separation

## Overview

The system now supports running OpenRouter and Gemini models with optimal strategies for each provider:

- **OpenRouter**: Smart queue system with rate limiting (20 calls/minute)
- **Gemini**: Concurrent execution (no artificial rate limits)

## Local Execution

### Run Single Provider

```bash
# OpenRouter models only (10 models)
python3 main.py --all-models-smart --provider openrouter --batch-size 20

# Gemini models only (6 models)
python3 main.py --all-models-smart --provider gemini

# All models (not recommended - uses OpenRouter strategy for all)
python3 main.py --all-models-smart
```

### Run Both Providers in Parallel (Local)

```bash
# Terminal 1
python3 main.py --all-models-smart --provider openrouter --batch-size 20 &

# Terminal 2
python3 main.py --all-models-smart --provider gemini &

# Wait for both
wait
```

## SLURM Execution

### Option 1: Separate Jobs (Recommended)

```bash
# Submit both jobs in parallel
sbatch slurm_openrouter.sh
sbatch slurm_gemini.sh
```

### Option 2: Single Job with Parallel Tasks

```bash
# Submit single job that runs both providers
sbatch slurm_both_parallel.sh
```

### Option 3: Daily Automation

```bash
# Run 3 times per day
0 8,16,0 * * * cd /path/to/LLMCaller && sbatch slurm_openrouter.sh && sbatch slurm_gemini.sh
```

## Performance Comparison

| Provider | Models | Strategy | Expected Time |
|----------|--------|----------|---------------|
| OpenRouter | 10 | Smart Queue | ~60-90 minutes |
| Gemini | 6 | Concurrent | ~10-20 minutes |
| **Total** | **16** | **Parallel** | **~60-90 minutes** |

## Output Structure

Results are saved in the same format for both providers:

```
experiments/
├── config_llama-3.3-8b/          # OpenRouter model
│   └── run_20250921_185840/
├── config_gemini-2.5-pro/        # Gemini model
│   └── run_20250921_185840/
└── config_gemini-2.5-pro-grounded/  # Gemini grounded
    └── run_20250921_185840/
```

## Notes

- **Credentials**: Set in `config/llm_setup.json` under `global.credentials_dir`
- **Logs**: Check `logs/` directory for detailed execution logs
- **Batch Size**: Only affects OpenRouter models (Gemini ignores this parameter)
- **Rate Limits**: OpenRouter has 20 calls/minute, Gemini has much higher limits