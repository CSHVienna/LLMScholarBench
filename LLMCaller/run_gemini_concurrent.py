import asyncio
import os
from datetime import datetime
from config.loader import load_llm_setup, load_category_variables
from prompts.generator import generate_prompt
from api.api_factory import create_api_client
from validation.validator import ResponseValidator
from storage.saver import save_attempt
from storage.summarizer import update_summary
from logs.setup import setup_logging

async def run_single_gemini_experiment(model_name, category, variable, run_dir, logger):
    """Run a single experiment with full response saving and retry logic"""

    # Load model config
    config = load_llm_setup(model_name)

    # Create API client
    api_client = create_api_client(config)

    # Generate prompt
    prompt = generate_prompt(category, variable)

    # Retry logic
    max_attempts = config.get('max_attempts', 3)

    for attempt_number in range(1, max_attempts + 1):
        try:
            logger.info(f"🔄 {model_name}:{category}:{variable} - Attempt {attempt_number}")

            # Call Gemini API (returns full response)
            api_response = await api_client.generate_response(prompt)

            # Extract text for validation (handle both normal and grounded responses)
            if hasattr(api_response, 'candidates') and hasattr(api_response.candidates[0], 'content'):
                # Normal Vertex AI response
                response_text = api_response.candidates[0].content.parts[0].text
                api_response_dict = {
                    "model": config['model'],
                    "prompt": prompt,
                    "response": response_text,
                    "full_vertex_response": str(api_response),  # Full object as string
                    "provider": "gemini",
                    "grounded": config.get('grounded', False)
                }
            elif isinstance(api_response, dict) and 'candidates' in api_response:
                # Grounded API response (JSON)
                response_text = api_response['candidates'][0]['content']['parts'][0]['text']
                api_response_dict = {
                    "model": config['model'],
                    "prompt": prompt,
                    "response": response_text,
                    "full_api_response": api_response,  # Full JSON response
                    "provider": "gemini",
                    "grounded": config.get('grounded', True)
                }
            else:
                raise Exception("Unexpected API response format")

            # Validate response
            validator = ResponseValidator()
            is_valid, message, extracted_data = validator.validate_response(response_text, category)

            # Prepare result with full API response (same format as OpenRouter)
            result = {
                "category": category,
                "variable": variable,
                "prompt": prompt,
                "full_api_response": api_response_dict,
                "validation_result": {
                    "is_valid": is_valid,
                    "message": message,
                    "extracted_data": extracted_data
                },
                "attempt": attempt_number
            }

            # ALWAYS save result first regardless of validation
            result_path = save_attempt(result, run_dir)
            summary_path = update_summary(result, run_dir)

            if is_valid:
                logger.info(f"✅ {model_name}:{category}:{variable} - Success - {result_path}")
                return result
            else:
                logger.info(f"❌ {model_name}:{category}:{variable} - Invalid (attempt {attempt_number}) - {message}")
                if attempt_number == max_attempts:
                    logger.info(f"💀 {model_name}:{category}:{variable} - All attempts failed")
                    return result  # Return last attempt even if invalid
                # Continue to next attempt

        except Exception as e:
            logger.error(f"🚨 {model_name}:{category}:{variable} - Attempt {attempt_number} error: {e}")

            if attempt_number == max_attempts:
                # Save error result
                error_result = {
                    "category": category,
                    "variable": variable,
                    "prompt": prompt,
                    "error": str(e),
                    "validation_result": {
                        "is_valid": False,
                        "message": f"API error: {str(e)}",
                        "extracted_data": None
                    },
                    "attempt": attempt_number
                }
                result_path = save_attempt(error_result, run_dir)
                summary_path = update_summary(error_result, run_dir)
                logger.error(f"💀 {model_name}:{category}:{variable} - Final failure - {result_path}")
                return error_result

            # Wait a bit before retry (small delay for API stability)
            await asyncio.sleep(1)

async def run_gemini_concurrent(models, output_dir=None, category=None, variable=None):
    """Run all Gemini models sequentially (one experiment at a time)"""

    print(f"🧠 Running {len(models)} Gemini models sequentially")

    # Load experiments
    if category and variable:
        experiments = [(category, variable)]
    else:
        category_variables = load_category_variables()
        experiments = []
        for cat, variables in category_variables.items():
            for var in variables:
                experiments.append((cat, var))

    print(f"📊 Total experiments per model: {len(experiments)}")
    print(f"📊 Total tasks: {len(models)} models × {len(experiments)} experiments = {len(models) * len(experiments)}")

    all_results = []
    start_time = asyncio.get_event_loop().time()

    for model_name in models:
        # Create experiment config directory (same pattern as existing)
        if output_dir is None:
            from config.loader import get_global_config
            global_config = get_global_config()
            output_dir = global_config.get('output_dir', 'experiments')

        base_config_dir = os.path.join(output_dir, f"config_{model_name}")
        os.makedirs(base_config_dir, exist_ok=True)

        # Save model config
        config = load_llm_setup(model_name)
        config_file_path = os.path.join(base_config_dir, "llm_setup.json")
        if not os.path.exists(config_file_path):
            import json
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=2)

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_config_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Setup logger for this model with unique name
        logger = setup_logging(run_dir, model_name=model_name)

        # Run experiments sequentially for this model
        print(f"\n🚀 Running {len(experiments)} experiments for {model_name}...")
        for cat, var in experiments:
            result = await run_single_gemini_experiment(model_name, cat, var, run_dir, logger)
            all_results.append(result)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    # Summary
    success_count = sum(1 for r in all_results if isinstance(r, dict) and r.get('validation_result', {}).get('is_valid', False))
    error_count = sum(1 for r in all_results if isinstance(r, Exception))

    print(f"\n🎉 Gemini sequential execution completed!")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Average per task: {total_time/len(all_results):.1f} seconds")
    print(f"   Valid results: {success_count}/{len(all_results)}")
    print(f"   Errors: {error_count}")

    return all_results