import time
from datetime import datetime
from config.loader import load_category_variables
from config.validator import validate_llm_setup
from logs.setup import setup_logging
from storage.saver import save_attempt
from storage.summarizer import update_summary
from api.groq_api import GroqAPI
from validation.validator import ResponseValidator
from prompts.generator import generate_prompt
import random

class ExperimentRunner:
    def __init__(self, run_dir, config, api_key, discipline="physics"):
        self.run_dir = run_dir
        self.discipline = discipline
        self.logger = setup_logging(run_dir)
        self.config = self._validate_config(config)
        self.api_client = GroqAPI(api_key, self.config)
        self.validator = ResponseValidator()

    def _validate_config(self, config):
        validate_llm_setup(config)
        return config

    def run_experiment(self):
        self.logger.info(f"Experiment run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for discipline: {self.discipline}")
        
        categories_variables = load_category_variables()
        all_pairs = [(category, variable) for category, variables in categories_variables.items() for variable in variables]
        random.shuffle(all_pairs)

        for category, variable in all_pairs:
            self.run_variable_experiment(category, variable)

        self.logger.info(f"Experiment run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



    def run_variable_experiment(self, category, variable):
        max_attempts = self.config.get('max_attempts', 3)
        prompt = generate_prompt(category, variable, self.discipline)

        for attempt in range(1, max_attempts + 1):
            try:
                api_response = self.api_client.generate_response(prompt)
                response_content = api_response.choices[0].message.content
                is_valid, message, extracted_data = self.validator.validate_response(response_content)

                result = {
                    "category": category,
                    "variable": variable,
                    "discipline": self.discipline,
                    "prompt": prompt,
                    "full_api_response": api_response.model_dump(),
                    "validation_result": {
                        "is_valid": is_valid,
                        "message": message,
                        "extracted_data": extracted_data
                    },
                    "attempt": attempt
                }

                result_path = save_attempt(result, self.run_dir)
                summary_path = update_summary(result, self.run_dir)
                self.logger.info(f"Result saved for {category}: {variable} (discipline: {self.discipline}) - Attempt {attempt}/{max_attempts} - Path: {result_path}")
                self.logger.info(f"Summary updated: {summary_path}")

                if is_valid:
                    break
            except Exception as e:
                self.logger.error(f"Error for {category}: {variable} (discipline: {self.discipline}) - Attempt {attempt}/{max_attempts} - {str(e)}")
                error_result = {
                    "category": category,
                    "variable": variable,
                    "discipline": self.discipline,
                    "attempt": attempt,
                    "error": {
                        "error_type": type(e).__name__,
                        "message": str(e)
                    }
                }
                result_path = save_attempt(error_result, self.run_dir)
                summary_path = update_summary(error_result, self.run_dir)
                self.logger.info(f"Error result saved for {category}: {variable} (discipline: {self.discipline}) - Attempt {attempt}/{max_attempts} - Path: {result_path}")

            if attempt < max_attempts:
                time.sleep(5)

    def run_specific_category(self, category):
        self.logger.info(f"Running experiments for category: {category}")
        categories_variables = load_category_variables()
        
        if category not in categories_variables:
            self.logger.error(f"Category '{category}' not found")
            return
        
        for variable in categories_variables[category]:
            self.run_variable_experiment(category, variable)

    def run_specific_parameter(self, category, variable):
        self.logger.info(f"Running experiment for parameter: {category}:{variable}")
        self.run_variable_experiment(category, variable)
