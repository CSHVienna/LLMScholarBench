#!/usr/bin/env python3
"""
Simplified async runner for OpenRouter models (paid tier - no rate limiting).
Runs all models × all experiments in parallel.
"""

import asyncio
import os
import json
from datetime import datetime
from config.loader import load_category_variables, load_llm_setup, get_global_config
from api.openrouter_api_async import OpenRouterAPIAsync
from validation.validator import ResponseValidator
from prompts.generator import generate_prompt
from storage.saver import save_attempt
from storage.summarizer import update_summary
from logs.setup import setup_logging

class OpenRouterAsyncRunner:
    """
    Simplified async runner for paid OpenRouter models.
    Runs all models × all experiments in parallel (like temp_validator.py).
    """

    def __init__(self, output_dir=None, temperature_override=None):
        self.output_dir = output_dir or get_global_config().get('output_dir', 'experiments')
        self.temperature_override = temperature_override
        self.validator = ResponseValidator()
        # Cache run directories and loggers per model (created once per model)
        self.model_run_dirs = {}
        self.model_loggers = {}
        # Global logger for the entire run (all models)
        self.global_logger = None
        self.global_run_timestamp = None

    def _create_global_logger(self):
        """Create global logger for the entire run (all models)"""
        if self.global_logger is None:
            # Create timestamp for this run
            self.global_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create global log file in output directory
            global_log_dir = os.path.join(self.output_dir, "global_runs")
            os.makedirs(global_log_dir, exist_ok=True)

            global_log_file = os.path.join(global_log_dir, f"run_{self.global_run_timestamp}.log")

            # Create logger
            import logging
            self.global_logger = logging.getLogger(f"global_run_{self.global_run_timestamp}")
            self.global_logger.setLevel(logging.INFO)

            # File handler
            file_handler = logging.FileHandler(global_log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.global_logger.addHandler(file_handler)

            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.global_logger.addHandler(console_handler)

        return self.global_logger

    def _get_or_create_run_dir(self, model_name):
        """Get or create output directory for this model run (once per model)"""
        if model_name not in self.model_run_dirs:
            # Use global timestamp if available, otherwise create new one
            if self.global_run_timestamp:
                timestamp = self.global_run_timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create base directory for this model configuration
            base_config_dir = os.path.join(self.output_dir, f"config_{model_name}")
            os.makedirs(base_config_dir, exist_ok=True)

            # Create a new directory for this run
            run_dir = os.path.join(base_config_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            # Cache it
            self.model_run_dirs[model_name] = run_dir
            # Create logger once per model (pass model_name to avoid logger collision)
            self.model_loggers[model_name] = setup_logging(run_dir, model_name)

        return self.model_run_dirs[model_name], self.model_loggers[model_name]

    async def run_single_experiment(self, model_name, category, variable, temperature_override=None):
        """
        Run a single experiment (one model, one category-variable pair).
        Two-level retry strategy:
        - Infrastructure retries (5x): for API failures (429, JSONDecodeError, BadRequest)
        - Validation retries (3x): for invalid responses (wrong format, schema errors)
        """
        # Load config and setup
        config = load_llm_setup(model_name)

        # Override temperature if provided
        if temperature_override is not None:
            max_temp = config.get('max_temperature', 2)
            if max_temp == 1:
                actual_temp = temperature_override / 2
            else:
                actual_temp = temperature_override
            config['temperature'] = actual_temp

        # Get run directory and logger (created once per model, shared across experiments)
        run_dir, logger = self._get_or_create_run_dir(model_name)

        # Create API client
        api_client = OpenRouterAPIAsync(config)

        # Generate prompt
        prompt = generate_prompt(category, variable)

        # Two-level retry limits
        max_validation_attempts = config.get('max_attempts', 3)
        max_infra_retries = 5

        # Validation attempt loop (for response quality issues)
        for validation_attempt in range(1, max_validation_attempts + 1):
            api_response = None
            infra_errors = []  # Track all infrastructure errors for this validation attempt

            # Infrastructure retry loop (for API failures)
            for infra_retry in range(max_infra_retries):
                try:
                    # Call API
                    api_response = await api_client.generate_response(prompt)

                    # Success! We got a response from the API
                    break  # Exit infrastructure retry loop

                except Exception as e:
                    error_name = type(e).__name__

                    # Record this infrastructure error
                    infra_errors.append({
                        "retry_number": infra_retry + 1,
                        "error_type": error_name,
                        "message": str(e)
                    })

                    # Check if this is an infrastructure error that should be retried
                    is_infra_error = any(err in error_name for err in [
                        'RateLimitError', 'BadRequestError', 'JSONDecodeError',
                        'APIConnectionError', 'APITimeoutError', 'InternalServerError'
                    ])

                    # Also check error message for specific patterns
                    error_msg = str(e).lower()
                    is_infra_error = is_infra_error or any(pattern in error_msg for pattern in [
                        'rate limit', '429', 'timeout', 'connection', 'expecting value'
                    ])

                    if is_infra_error and infra_retry < max_infra_retries - 1:
                        # Infrastructure error - retry with exponential backoff
                        wait_time = min(5 * (2 ** infra_retry), 60)  # 5s, 10s, 20s, 40s, 60s
                        logger.warning(
                            f"🔄 API infrastructure error ({error_name}): {model_name}:{category}:{variable} - "
                            f"Infra retry {infra_retry + 1}/{max_infra_retries} after {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Either non-infra error OR max infra retries reached
                        logger.error(
                            f"❌ API error after {infra_retry + 1} infrastructure retries: "
                            f"{model_name}:{category}:{variable} - {error_name}: {str(e)}"
                        )

                        # Save permanent infrastructure failure
                        error_result = {
                            "category": category,
                            "variable": variable,
                            "prompt": prompt,
                            "attempt": validation_attempt,
                            "infrastructure_retries": infra_retry + 1,
                            "infrastructure_errors": infra_errors,  # All errors from all retries
                            "error": {
                                "error_type": error_name,
                                "message": str(e)
                            },
                            "validation_result": {
                                "is_valid": False,
                                "message": "Infrastructure failure - API call failed",
                                "extracted_data": None
                            },
                            "full_api_response": {
                                "error_from_exception": str(e),
                                "exception_type": error_name,
                                "note": "API call itself failed after infrastructure retries"
                            }
                        }

                        result_path = save_attempt(error_result, run_dir)
                        update_summary(error_result, run_dir)

                        # Continue to next validation attempt (don't give up yet)
                        if validation_attempt < max_validation_attempts:
                            logger.info(
                                f"🔄 Infrastructure failure on attempt {validation_attempt}/{max_validation_attempts}, "
                                f"will retry with fresh infrastructure retries"
                            )
                            await asyncio.sleep(2)
                            continue
                        else:
                            return error_result  # Max validation attempts reached

            # If we got here without api_response, something unexpected happened
            if api_response is None:
                logger.error(f"❌ Unexpected: No API response after infrastructure retries")
                continue

            # Check if API response contains an error (e.g., 502 from provider)
            if hasattr(api_response, 'error') and api_response.error is not None:
                error_code = api_response.error.get('code', 'unknown') if isinstance(api_response.error, dict) else 'unknown'
                error_msg = api_response.error.get('message', str(api_response.error)) if isinstance(api_response.error, dict) else str(api_response.error)

                # Record this as an infrastructure error
                infra_errors.append({
                    "retry_number": len(infra_errors) + 1,
                    "error_type": f"APIErrorResponse_{error_code}",
                    "message": error_msg
                })

                logger.warning(
                    f"🔄 API returned error response ({error_code}): {model_name}:{category}:{variable} - "
                    f"Treating as infrastructure error, will retry validation attempt"
                )

                # Save this error attempt and continue to next validation attempt
                error_result = {
                    "category": category,
                    "variable": variable,
                    "prompt": prompt,
                    "attempt": validation_attempt,
                    "infrastructure_errors": infra_errors,
                    "full_api_response": api_response.model_dump() if hasattr(api_response, 'model_dump') else {"error": api_response.error},
                    "validation_result": {
                        "is_valid": False,
                        "message": f"API returned error response: {error_msg}",
                        "extracted_data": None
                    }
                }

                result_path = save_attempt(error_result, run_dir)
                update_summary(error_result, run_dir)

                # Continue to next validation attempt (don't give up yet)
                if validation_attempt < max_validation_attempts:
                    await asyncio.sleep(2)
                    continue
                else:
                    return error_result  # Max validation attempts reached

            # We have an API response - now validate it
            try:
                response_content = api_response.choices[0].message.content
                is_valid, message, extracted_data = self.validator.validate_response(response_content, category)

                # Prepare result (always include full API response)
                result = {
                    "category": category,
                    "variable": variable,
                    "prompt": prompt,
                    "full_api_response": api_response.model_dump(),
                    "validation_result": {
                        "is_valid": is_valid,
                        "message": message,
                        "extracted_data": extracted_data
                    },
                    "attempt": validation_attempt
                }

                # ALWAYS save result first (regardless of validation)
                result_path = save_attempt(result, run_dir)
                update_summary(result, run_dir)
                logger.info(
                    f"✅ Saved result for {model_name}:{category}:{variable} - "
                    f"Validation attempt {validation_attempt}/{max_validation_attempts} - Valid: {is_valid}"
                )

                if is_valid:
                    return result  # Success!
                else:
                    # Validation failed - retry if not max attempts
                    if validation_attempt < max_validation_attempts:
                        logger.info(
                            f"⚠️  Validation failed, will retry: {model_name}:{category}:{variable} - "
                            f"{message} (Attempt {validation_attempt}/{max_validation_attempts})"
                        )
                        await asyncio.sleep(2)  # Brief pause before next validation attempt
                        continue
                    else:
                        logger.warning(
                            f"❌ Max validation attempts reached: {model_name}:{category}:{variable} - {message}"
                        )
                        return result  # Return last attempt

            except Exception as e:
                # Processing error (API succeeded but we failed to process the response)
                logger.error(
                    f"❌ Processing error for {model_name}:{category}:{variable} - "
                    f"Attempt {validation_attempt}/{max_validation_attempts} - {str(e)}"
                )

                error_result = {
                    "category": category,
                    "variable": variable,
                    "prompt": prompt,
                    "attempt": validation_attempt,
                    "full_api_response": api_response.model_dump() if api_response else {},
                    "processing_error": {
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "note": "API call succeeded but processing failed"
                    },
                    "validation_result": {
                        "is_valid": False,
                        "message": "Processing error occurred",
                        "extracted_data": None
                    }
                }

                # Save error result
                result_path = save_attempt(error_result, run_dir)
                update_summary(error_result, run_dir)

                # Retry if not max attempts
                if validation_attempt < max_validation_attempts:
                    await asyncio.sleep(2)
                    continue
                else:
                    return error_result  # Return error on final attempt

    async def run_all_models(self, models, category=None, variable=None):
        """
        Run all OpenRouter models asynchronously (all models × all experiments in parallel).

        Args:
            models: List of model names to run
            category: Optional single category to run
            variable: Optional single variable to run (requires category)
        """
        # Create global logger first
        global_logger = self._create_global_logger()

        global_logger.info("=" * 80)
        global_logger.info(f"🚀 Starting async OpenRouter run with {len(models)} models")
        global_logger.info(f"   Mode: Full async parallelization (no rate limiting)")
        global_logger.info(f"   Timestamp: {self.global_run_timestamp}")
        global_logger.info(f"   Models: {', '.join(models)}")

        # Load experiments
        if category and variable:
            experiments = [(category, variable)]
        else:
            categories_variables = load_category_variables()
            experiments = [(cat, var) for cat, variables in categories_variables.items() for var in variables]

        total_experiments = len(models) * len(experiments)
        global_logger.info(f"   Total experiments: {total_experiments} ({len(models)} models × {len(experiments)} experiments)")

        # Create all tasks (all models × all experiments)
        all_tasks = []
        for model in models:
            for exp_category, exp_variable in experiments:
                task = self.run_single_experiment(
                    model,
                    exp_category,
                    exp_variable,
                    self.temperature_override
                )
                all_tasks.append(task)

        global_logger.info(f"   🔄 Running {len(all_tasks)} parallel requests...")
        global_logger.info("=" * 80)

        # Run ALL tasks in parallel
        start_time = datetime.now()
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        end_time = datetime.now()

        # Calculate statistics
        duration = (end_time - start_time).total_seconds()
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('validation_result', {}).get('is_valid', False))
        failed = len(results) - successful

        global_logger.info("=" * 80)
        global_logger.info(f"🎉 Async OpenRouter run completed!")
        global_logger.info(f"   Total experiments: {total_experiments}")
        global_logger.info(f"   Successful: {successful}")
        global_logger.info(f"   Failed: {failed}")
        global_logger.info(f"   Duration: {duration/60:.1f} minutes")
        global_logger.info(f"   Average: {duration/total_experiments:.2f}s per experiment")

        # Summary per model
        global_logger.info("")
        global_logger.info("📊 Per-Model Summary:")
        for model in models:
            # Filter results for this model (handle None values safely)
            model_results = [
                r for r in results
                if isinstance(r, dict) and
                r.get('full_api_response', {}).get('model') and
                r.get('full_api_response', {}).get('model', '').endswith(model)
            ]
            if model_results:
                model_success = sum(1 for r in model_results if r.get('validation_result', {}).get('is_valid', False))
                global_logger.info(f"   {model}: {model_success}/{len(model_results)} successful")

        global_logger.info("=" * 80)

        return {
            'results': results,
            'stats': {
                'total_experiments': total_experiments,
                'successful': successful,
                'failed': failed,
                'duration_seconds': duration,
                'duration_minutes': duration / 60
            }
        }
