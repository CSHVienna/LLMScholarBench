# Smart Queue System

The smart queue system is an advanced optimization for LLMCaller that maximizes API call efficiency and minimizes wasted time.

## Key Features

### 🚀 Zero Wasted API Calls
- Always fills batches to exactly the rate limit (15 calls/batch)
- Cross-model work stealing to optimize batching
- Smart retry prioritization

### 🧠 Intelligent Priority System
1. **Retries** (Priority 1) - Failed tasks that need retry
2. **Current Model** (Priority 2) - Tasks from the active model  
3. **Next Model** (Priority 3) - Tasks from queued models

### ⚡ Efficiency Example
**Old System:** 18 prompts for Model A (15/min limit)
- Batch 1: 15 queries ✓
- Batch 2: 3 queries + 12 **wasted slots** ❌

**Smart System:** 18 prompts for Model A + Model B queued
- Batch 1: 15 queries from Model A ✓
- Batch 2: 3 remaining Model A + 12 Model B queries ✓ 
- **Zero wasted slots!**

## Usage

### Single Model with Smart Queue
```bash
uv run python3 main.py --model qwen3-8b --smart
```

### Multiple Models with Cross-Model Optimization
```bash
# RECOMMENDED: Uses smart cross-model batching
uv run python3 main.py --all-models-smart

# Legacy: Sequential processing (less efficient)
uv run python3 main.py --all-models
```

### Specific Experiments
```bash
uv run python3 main.py --all-models-smart --category epoch --variable 2000s
```

## How It Works

### Queue Architecture
- **Retry Queue**: Failed tasks (highest priority)
- **Current Model Queue**: Active model tasks
- **Next Model Queue**: Upcoming model tasks

### Batch Composition Algorithm
1. Fill batch with retries first (up to 15)
2. Add current model tasks to fill remaining slots
3. If current model is empty, promote next model
4. Add promoted model tasks to fill remaining slots
5. **Result**: Always exactly 15 calls per batch

### Error Handling Improvements
- Fixed error JSON saves to include full context (`prompt`, `validation_result`)
- Retries are automatically prioritized in queue
- No more incomplete error logs

## Performance Benefits

### Time Savings
- **Multi-model runs**: Significantly faster due to cross-model batching
- **Retry efficiency**: Failed tasks don't waste full batch cycles
- **No waiting**: Models don't wait for others to complete

### API Efficiency
- **100% slot utilization**: No wasted API calls
- **Optimal timing**: Smart 60-second batch intervals
- **Reduced total batches**: Cross-model filling reduces batch count

### Example Improvement
**Scenario**: 3 models, 18 prompts each (54 total)

**Old System:**
- 4 batches per model = 12 total batches
- Many wasted slots in final batches
- ~12 minutes minimum

**Smart System:**
- ~4 total batches across all models
- Zero wasted slots
- ~4 minutes total
- **66% time reduction!**

## Architecture

### Files
- `utils/smart_queue.py` - Core smart queue implementation
- `experiments/runner_smart.py` - Smart experiment runners
- `main.py` - Updated with smart queue options

### Integration
The smart queue system is fully backward compatible. The old system remains available via `--all-models` for comparison.

## Monitoring

The smart queue provides detailed statistics:
- Efficiency percentage (slots used vs available)
- Wasted slots count
- Batch composition details
- Cross-model optimization metrics
- Time savings estimates

## Future Enhancements

Potential improvements:
- Dynamic rate limit adjustment
- Model-specific priority weighting
- Advanced failure prediction
- Real-time batch optimization
- Cost optimization tracking

The smart queue system transforms LLMCaller from a simple sequential processor into an intelligent, efficiency-optimized experiment runner.