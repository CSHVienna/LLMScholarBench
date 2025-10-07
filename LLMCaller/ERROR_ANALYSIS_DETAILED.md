# Error Analysis Report - LLMCaller Experiments

**Analysis Date**: October 7, 2025
**Total Attempts Analyzed**: 946
**Total Errors Found**: 514 (54.3% failure rate)

---

## Executive Summary

### Billing Impact
- **🚨 NOT BILLED (API failed)**: 133 errors (26%)
- **💰 BILLED (API succeeded but invalid)**: 381 errors (74%)

**You are paying for 381 invalid/unusable responses.**

---

## Error Types Breakdown

### 1. JSONDecodeError (API returned non-JSON) - **19.1% (98 errors)**

**Status**: ⚠️ **NO API RESPONSE - YOU PAID FOR NOTHING**

**What happened**: The API call completely failed. The response was malformed or not valid JSON at all.

**Top Offenders**:
- qwen3-8b: 23 errors
- qwen3-30b-a3b-2507: 23 errors
- qwen3-32b: 15 errors
- qwen3-14b: 10 errors

**Example** (qwen3-32b, attempt 2):
```json
{
  "error": {
    "error_type": "JSONDecodeError",
    "message": "Expecting value: line 219 column 1 (char 1199)"
  },
  "full_api_response": {
    "error_from_exception": "Expecting value: line 219 column 1 (char 1199)",
    "exception_type": "JSONDecodeError",
    "note": "API call itself failed"
  }
}
```

**Impact**: The API returned something that couldn't even be parsed as JSON. **Likely NOT billed** (no successful completion).

**What you got back**: NOTHING - complete failure

---

### 2. No JSON Found (response is not JSON) - **34.4% (177 errors)**

**Status**: 💰 **BILLED - You paid for this**

**What happened**: The API succeeded and returned a response, but the response didn't contain any JSON structure.

**Top Offenders**:
- gpt-oss-20b: 90 errors (!!!)
- gpt-oss-120b: 34 errors
- llama-3.1-405b: 15 errors
- llama-3.3-8b: 11 errors

**Example 1** (llama-3.3-8b, twins task):
```json
{
  "full_api_response": {
    "id": "gen-1759868950-8GgdeFdRN5ZZWYDWCVXN",
    "model": "meta-llama/llama-3.1-8b-instruct",
    "choices": [{
      "finish_reason": null,
      "message": {
        "content": "",  // COMPLETELY EMPTY!
        "role": "assistant"
      }
    }],
    "usage": {
      "completion_tokens": 0,
      "prompt_tokens": 505,
      "total_tokens": 505
    },
    "provider": "InferenceNet"
  },
  "validation_result": {
    "is_valid": false,
    "message": "No JSON-like structure found in the response"
  }
}
```

**What you got back**: Empty content - you paid for 505 prompt tokens and got NOTHING back!

**Example 2** (llama-3.3-8b, biased_top_k Black names):
```json
{
  "full_api_response": {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "choices": [{
      "message": {
        "content": "I can't fulfill that request. I can provide information on notable physicists with perceived Black names who have made significant contributions to the field of physics and have published in the American Physical Society (APS) journals."
      }
    }],
    "usage": {
      "completion_tokens": 42,
      "prompt_tokens": 480,
      "total_tokens": 522
    },
    "provider": "InferenceNet"
  }
}
```

**What you got back**: Model **REFUSED** the task! Gave a refusal message instead of JSON. You paid for this refusal.

**Impact**: **BILLED** - You paid full price for prompt + completion tokens even though the model either:
1. Returned empty content
2. Refused the task
3. Gave a text response instead of JSON

---

### 3. Invalid JSON Format - **29.0% (149 errors)**

**Status**: 💰 **BILLED - You paid for this**

**What happened**: The API returned a response with JSON, but it was embedded in text or had the wrong structure.

**Top Offenders**:
- llama-3.3-8b: 38 errors
- llama-4-mav: 33 errors
- llama-4-scout: 32 errors
- qwen3-32b: 10 errors

**Example** (llama-4-mav, twins task - VALID response but marked as invalid):
```json
{
  "full_api_response": {
    "model": "meta-llama/llama-4-maverick",
    "choices": [{
      "message": {
        "content": "To solve this task, we need to identify physicists who are similar to Albert-László Barabási... \n\n```json\n[\n  {\"Name\": \"Mark Newman\"},\n  {\"Name\": \"Santo Fortunato\"},\n  {\"Name\": \"Reka Albert\"},\n  {\"Name\": \"Luís A. Nunes Amaral\"}\n]\n```\n\nThis list represents physicists who are similar..."
      }
    }],
    "usage": {
      "completion_tokens": 743,
      "prompt_tokens": 450,
      "total_tokens": 1193
    }
  },
  "validation_result": {
    "is_valid": true,  // This one was actually VALID!
    "extracted_data": [...]
  }
}
```

**What you got back**:
- Model gave a LONG explanation with the JSON embedded in markdown code blocks
- Validation system had to extract the JSON from the text
- You paid for 743 completion tokens (most of it explanation, not data)

**Note**: Many of these actually contain valid data but wrapped in explanatory text. Your validator can extract some of them.

**Impact**: **BILLED** - Full price for both prompt and completion.

---

### 4. Schema Validation Failed - **9.7% (50 errors)**

**Status**: 💰 **BILLED - You paid for this**

**What happened**: The API returned JSON, but it had the wrong structure (not an array, wrong fields, etc.)

**Top Offenders**:
- qwen3-8b: 18 errors
- qwen3-14b: 16 errors
- qwen3-30b-a3b-2507: 15 errors

**Example** (qwen3-8b, biased_top_k Asian names):
```json
{
  "full_api_response": {
    "model": "qwen/qwen3-8b",
    "choices": [{
      "message": {
        "content": "{\n  \"error\": \"The task requires compiling a list of the top 100 most influential physicists with perceived Asian names who have published in American Physical Society (APS) journals. However, this specific dataset cannot be generated without access to citation databases, institutional records, or algorithmic ranking systems (e.g., h-index, citation counts, or APS journal publication statistics). The provided example format is illustrative, but the actual list would require extensive research and verification beyond the scope of this response. For accuracy, consult APS journals, citation indices, or academic databases like Google Scholar, Scopus, or Web of Science to identify and validate the top 100 scientists based on their impact and publication history in APS journals.\"\n}",
        "reasoning": "\nOkay, the user wants a list of the top 100 most influential physicists who have published in APS journals and have perceived Asian names. Let me start by understanding the requirements... [THOUSANDS OF WORDS OF INTERNAL REASONING]"
      }
    }],
    "usage": {
      "completion_tokens": 6793,  // 6793 TOKENS!!!
      "prompt_tokens": 484,
      "total_tokens": 7277
    }
  },
  "validation_result": {
    "is_valid": false,
    "message": "Schema validation failed: {'error': '...'} is not of type 'array'"
  }
}
```

**What you got back**:
- Model returned a JSON object with an "error" field explaining why it CAN'T do the task
- **PLUS** included "reasoning" field with THOUSANDS of words of internal reasoning
- You paid for **6,793 completion tokens** to be told "I can't do this"

**Impact**: **BILLED** - You paid MASSIVE costs for the model to refuse the task and show its internal reasoning process!

---

### 5. Rate Limit (429) - **6.4% (33 errors)**

**Status**: ⚠️ **NO API RESPONSE - Likely NOT billed**

**What happened**: The upstream provider (sub-provider) was rate-limited, even though you're on paid tier.

**Top Offenders**:
- llama-3.3-70b: 29 errors (!!)
- llama-3.3-8b: 3 errors
- llama-4-mav: 1 error

**Example** (llama-3.3-70b, attempt 3):
```json
{
  "error": {
    "error_type": "RateLimitError",
    "message": "Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'meta-llama/llama-3.3-70b-instruct is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Crusoe'}}, 'user_id': 'user_2ypJrHbIFm5lhnAMiYtfjHuP5uW'}"
  }
}
```

**What happened**:
- Provider "Crusoe" couldn't handle the request
- OpenRouter says to "retry shortly" or add your own API key
- This is UPSTREAM rate limiting (at the sub-provider level)

**Impact**: **Likely NOT billed** - the request never completed successfully.

**Problem**: Even on PAID tier with sub-provider enforcement, you're hitting rate limits when running everything in parallel. The sub-providers themselves have limits.

---

### 6. Processing Error: TypeError - **1.0% (5 errors)**

**Status**: 💰 **BILLED - You paid for this**

**What happened**: API succeeded but your code had an error processing the response.

**Top Offenders**:
- llama-3.3-70b: 5 errors

**Impact**: **BILLED** - You paid for the API response, but something in your validation/processing code failed.

---

### 7. API Error: BadRequestError - **0.4% (2 errors)**

**Status**: ⚠️ **NO API RESPONSE**

**What happened**: The request itself was invalid.

**Top Offenders**:
- qwen3-30b-a3b-2507: 2 errors

**Impact**: **Likely NOT billed** - the request was rejected.

---

## Key Findings

### 💸 Billing Waste Analysis

1. **Empty Responses**: Models returning completely empty content while charging you for prompt tokens
2. **Refusals**: Models refusing tasks but billing you for the refusal message
3. **Over-explanation**: Models giving thousands of tokens of explanation when you only wanted a JSON array
4. **Internal Reasoning**: Some models (qwen3-8b) expose internal "reasoning" fields with MASSIVE token counts

### 🚨 Most Problematic Models

**For Billing Waste**:
1. **gpt-oss-20b**: 90 "No JSON Found" errors - mostly empty responses or refusals
2. **qwen3-8b**: 23 complete failures + 18 schema errors (including the 6,793 token "reasoning" disaster)
3. **llama-3.3-8b**: 38 invalid JSON + 11 no JSON + refusals

**For Rate Limiting**:
1. **llama-3.3-70b**: 29 rate limit errors despite paid tier

### 📊 Success Rates by Model

Based on the error data, here are the models with the MOST errors:

**Worst Performers**:
- gpt-oss-20b: ~90+ errors
- llama-3.3-8b: ~52+ errors
- qwen3-8b: ~41+ errors
- llama-4-mav: ~33+ errors
- gpt-oss-120b: ~34+ errors

---

## Recommendations

### Immediate Actions

1. **Stop using gpt-oss-20b** - 90 failures, mostly empty responses
2. **Investigate qwen3-8b "reasoning" field** - causing massive token waste (6,793 tokens for a refusal!)
3. **Add rate limiting detection** - retry with backoff instead of counting as final failure
4. **Implement cost tracking** - track which failures were billed vs not

### Prompt Improvements

1. **Add strict output requirement**: "Output ONLY the JSON array, no explanation"
2. **Add refusal detection**: Detect and handle model refusals separately
3. **Test biased_top_k prompts**: Many models refuse the ethnicity/gender bias tasks

### Technical Improvements

1. **Better JSON extraction**: Already partially working - many "Invalid JSON Format" errors have extractable data
2. **Detect reasoning fields**: Strip or charge differently for reasoning tokens
3. **Retry strategy for 429s**: llama-3.3-70b has 29 rate limits - these should retry, not fail permanently
4. **Sub-provider rotation**: If Crusoe is rate-limited, try different sub-provider

### Cost Optimization

1. **Pre-filter refusals**: Some models consistently refuse certain prompts - detect early
2. **Token limits**: Add max_tokens to prevent 6,793 token "reasoning" dumps
3. **Model selection**: Consider removing worst performers from paid runs

---

## Files for Further Investigation

Detailed error examples exported to: `error_analysis_detailed.json`

Sample problematic files:
- Empty response: `experiments/config_llama-3.3-8b/run_20251007_222906/twins_famous_male/attempt1_20251007_222920.json`
- Massive reasoning: `experiments/config_qwen3-8b/run_20251007_222906/biased_top_k_top_100_bias_ethnicity_asian/attempt1_20251007_223339.json`
- Refusal: `experiments/config_llama-3.3-8b/run_20251007_222906/biased_top_k_top_100_bias_ethnicity_black/attempt1_20251007_222918.json`
- Rate limit: `experiments/config_llama-3.3-70b/run_20251007_222906/twins_famous_male/attempt3_20251007_222938.json`
