import time
import asyncio
from datetime import datetime
from config.loader import load_category_variables
from config.validator import validate_llm_setup
from logs.setup import setup_logging
from storage.saver import save_attempt
from storage.summarizer import update_summary
from api.openrouter_api import OpenRouterAPI
from validation.validator import ResponseValidator
from prompts.generator import generate_prompt
import random

class ExperimentRunner:
    def __init__(self, run_dir, config):
        self.run_dir = run_dir
        self.logger = setup_logging(run_dir)
        self.config = self._validate_config(config)
        self.api_client = OpenRouterAPI(self.config)
        self.validator = ResponseValidator()

    def _validate_config(self, config):
        validate_llm_setup(config)
        return config

    def run_experiment(self):
        """Run experiment synchronously - wrapper for async implementation"""
        asyncio.run(self._run_experiment_async())
    
    async def _run_experiment_async(self):
        self.logger.info(f"Experiment run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        categories_variables = load_category_variables()
        all_pairs = [(category, variable) for category, variables in categories_variables.items() for variable in variables]
        random.shuffle(all_pairs)

        # Process experiments in batches of 20 (rate limit)
        batch_size = 20
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_pairs) + batch_size - 1)//batch_size} ({len(batch)} experiments)")
            
            # Run batch concurrently
            tasks = [self.run_variable_experiment_async(category, variable) for category, variable in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Small delay between batches to be safe
            if i + batch_size < len(all_pairs):
                await asyncio.sleep(1)

        # Print usage summary at the end
        usage_summary = self.api_client.get_usage_summary()
        self.logger.info(f"Usage Summary: {usage_summary}")
        print(f"\nUsage Summary: {usage_summary}")
        
        self.logger.info(f"Experiment run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def run_single_experiment(self, category, variable):
        """Run experiment for a single category-variable pair"""
        asyncio.run(self._run_single_experiment_async(category, variable))
    
    async def _run_single_experiment_async(self, category, variable):
        """Run experiment for a single category-variable pair"""
        self.logger.info(f"Single experiment started for {category}: {variable} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            try:
                # Call the API with the prompt
                api_response = await self.api_client.generate_response(prompt)
                
                # Validate the response
                response_content = api_response.choices[0].message.content
                is_valid, message, extracted_data = self.validator.validate_response(response_content, category)

                # Prepare the result for saving
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

                # Save the result
                result_path = save_attempt(result, self.run_dir)
                summary_path = update_summary(result, self.run_dir)
                self.logger.info(f"Result saved for {category}: {variable} - Attempt {attempt}/{max_attempts} - Path: {result_path}")
                self.logger.info(f"Summary updated: {summary_path}")

                if is_valid:
                    break  # Stop attempts if a valid result is obtained
            except Exception as e:
                self.logger.error(f"Error for {category}: {variable} - Attempt {attempt}/{max_attempts} - {str(e)}")
                error_result = {
                    "category": category,
                    "variable": variable,
                    "attempt": attempt,
                    "error": {
                        "error_type": type(e).__name__,
                        "message": str(e)
                    }
                }
                result_path = save_attempt(error_result, self.run_dir)
                summary_path = update_summary(error_result, self.run_dir)
                self.logger.info(f"Error result saved for {category}: {variable} - Attempt {attempt}/{max_attempts} - Path: {result_path}")

            if attempt < max_attempts:
                await asyncio.sleep(5)  # Wait for 5 seconds between attempts
