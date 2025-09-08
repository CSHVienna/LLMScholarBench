import time
import asyncio
from datetime import datetime
from config.loader import load_category_variables
from config.validator import validate_llm_setup
from logs.setup import setup_logging
from storage.saver import save_attempt
from storage.summarizer import update_summary
from api.api_factory import create_api_client
from validation.validator import ResponseValidator
from prompts.generator import generate_prompt
from usage.tracker import DailyUsageTracker
from utils.batching import BatchProcessor
import random

class ExperimentRunner:
    def __init__(self, run_dir, config, batch_size=15):
        self.run_dir = run_dir
        self.logger = setup_logging(run_dir)
        self.config = self._validate_config(config)
        self.api_client = create_api_client(self.config)
        self.validator = ResponseValidator()
        self.usage_tracker = DailyUsageTracker()
        self.batch_size = batch_size
        self.batch_processor = BatchProcessor(rate_limit=batch_size, logger=self.logger)

    def _validate_config(self, config):
        validate_llm_setup(config)
        return config

    def run_experiment(self):
        """Run experiment synchronously - wrapper for async implementation"""
        asyncio.run(self._run_experiment_async())
    
    async def _run_experiment_async(self):
        self.logger.info(f"Experiment run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load all experiments
        categories_variables = load_category_variables()
        all_pairs = [(category, variable) for category, variables in categories_variables.items() for variable in variables]
        random.shuffle(all_pairs)
        
        total_experiments = len(all_pairs)
        self.logger.info(f"Total experiments to run: {total_experiments}")
        
        # Pre-flight safety check
        can_run, usage_info = self.usage_tracker.can_run_experiments(total_experiments)
        if not can_run:
            error_msg = f"Cannot run {total_experiments} experiments - would exceed daily limit!"
            error_msg += f"\nCurrent usage: {usage_info['current_usage']}/{usage_info['daily_limit']}"
            error_msg += f"\nRemaining: {usage_info['remaining']}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        self.logger.info(f"Pre-flight check passed: {usage_info['current_usage']} + {total_experiments} = {usage_info['total_after']}/{usage_info['daily_limit']}")
        
        # Pre-flight rate limit check
        print("🔍 Checking current rate limit status...")
        rate_limit_ready = await self.api_client.wait_for_rate_limit_reset()
        if not rate_limit_ready:
            error_msg = "Pre-flight rate limit check failed - aborting experiments"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Process experiments using optimal batching
        results = await self.batch_processor.process_batches(
            all_pairs, 
            self._run_single_experiment_wrapper
        )
        
        # Record the run in usage tracker
        batch_stats = self.batch_processor.get_stats()
        await self.usage_tracker.record_run(
            calls_made=batch_stats['successful_experiments'],
            models_count=1,  # Single model per run
            experiments_count=total_experiments,
            metadata={
                'batch_size': 15,
                'total_batches': batch_stats['total_batches'],
                'success_rate': f"{batch_stats['successful_experiments']}/{total_experiments}",
                'total_time_minutes': round(batch_stats['total_time'] / 60, 2)
            }
        )
        
        # Print final summary
        usage_summary = self.api_client.get_usage_summary()
        self.logger.info(f"API Usage Summary: {usage_summary}")
        self.logger.info(f"Batch Processing Stats: {batch_stats}")
        
        print(f"\n🎉 Experiment run completed!")
        print(f"   Total experiments: {total_experiments}")
        print(f"   Successful: {batch_stats['successful_experiments']}")
        print(f"   Failed: {batch_stats['failed_experiments']}")
        print(f"   Total time: {batch_stats['total_time']/60:.1f} minutes")
        print(f"   API calls tracked: {usage_summary}")
        
        self.logger.info(f"Experiment run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    async def _run_single_experiment_wrapper(self, experiment_pair):
        """Wrapper for batch processor - runs a single experiment"""
        category, variable = experiment_pair
        try:
            await self.run_variable_experiment_async(category, variable)
            return f"Success: {category}:{variable}"
        except Exception as e:
            self.logger.error(f"Experiment failed {category}:{variable} - {str(e)}")
            raise e

    def run_single_experiment(self, category, variable):
        """Run experiment for a single category-variable pair"""
        asyncio.run(self._run_single_experiment_async(category, variable))
    
    async def _run_single_experiment_async(self, category, variable):
        """Run experiment for a single category-variable pair"""
        self.logger.info(f"Single experiment started for {category}: {variable} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Pre-flight rate limit check
        print("🔍 Checking current rate limit status...")
        rate_limit_ready = await self.api_client.wait_for_rate_limit_reset()
        if not rate_limit_ready:
            error_msg = "Pre-flight rate limit check failed - aborting experiment"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        await self.run_variable_experiment_async(category, variable)
        
        # Print usage summary
        usage_summary = self.api_client.get_usage_summary()
        self.logger.info(f"Usage Summary: {usage_summary}")
        print(f"\nUsage Summary: {usage_summary}")
        
        self.logger.info(f"Single experiment completed for {category}: {variable} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def run_variable_experiment(self, category, variable):
        """Synchronous wrapper for backward compatibility"""
        asyncio.run(self.run_variable_experiment_async(category, variable))

    async def run_variable_experiment_async(self, category, variable):
        """Run variable experiment asynchronously"""
        max_attempts = self.config.get('max_attempts', 3)
        prompt = generate_prompt(category, variable)

        for attempt in range(1, max_attempts + 1):
            api_response = None
            try:
                # Call the API with the prompt
                api_response = await self.api_client.generate_response(prompt)
                
                # Validate the response
                response_content = api_response.choices[0].message.content
                is_valid, message, extracted_data = self.validator.validate_response(response_content, category)

                # Prepare the result for saving (always include full API response)
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
                    "attempt": attempt
                }

                # ALWAYS save the result first (with full API response) regardless of validation
                result_path = save_attempt(result, self.run_dir)
                summary_path = update_summary(result, self.run_dir)
                self.logger.info(f"Result saved for {category}: {variable} - Attempt {attempt}/{max_attempts} - Path: {result_path}")
                self.logger.info(f"Summary updated: {summary_path}")

                if is_valid:
                    break  # Stop attempts if a valid result is obtained
                else:
                    # Validation failed - but we already saved the full response above
                    self.logger.info(f"❌ Invalid (will retry): {category}:{variable} - Attempt {attempt}/{max_attempts} - {message}")
                    # Don't throw exception here - let the retry loop handle it naturally
            except Exception as e:
                self.logger.error(f"Error for {category}: {variable} - Attempt {attempt}/{max_attempts} - {str(e)}")
                error_result = {
                    "category": category,
                    "variable": variable,
                    "prompt": prompt,
                    "attempt": attempt,
                    "error": {
                        "error_type": type(e).__name__,
                        "message": str(e)
                    },
                    "validation_result": {
                        "is_valid": False,
                        "message": "Error occurred during processing",
                        "extracted_data": None
                    }
                }
                
                # ALWAYS save whatever we got back - API response AND exception details
                if api_response is not None:
                    # API call succeeded, save the full response for debugging
                    error_result["full_api_response"] = api_response.model_dump()
                    # Also save what went wrong during processing
                    error_result["processing_error"] = {
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "note": "API call succeeded but processing failed (e.g., JSON parsing)"
                    }
                else:
                    # API call itself failed - save exception details
                    error_result["full_api_response"] = {
                        "error_from_exception": str(e),
                        "exception_type": type(e).__name__,
                        "raw_exception": str(e),
                        "note": "API call itself failed"
                    }
                
                result_path = save_attempt(error_result, self.run_dir)
                summary_path = update_summary(error_result, self.run_dir)
                self.logger.info(f"Error result saved for {category}: {variable} - Attempt {attempt}/{max_attempts} - Path: {result_path}")

            if attempt < max_attempts:
                await asyncio.sleep(5)  # Wait for 5 seconds between attempts
