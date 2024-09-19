import json
import re
import os
from jsonschema import validate, ValidationError

class ResponseValidator:
    def __init__(self, schema_dir='config/schemas'):
        self.schemas = self.load_schemas(schema_dir)

    def load_schemas(self, schema_dir):
        schemas = {}
        for filename in os.listdir(schema_dir):
            if filename.endswith('.json'):
                category = filename[:-5]  # Remove '.json' from the filename
                with open(os.path.join(schema_dir, filename), 'r') as f:
                    schemas[category] = json.load(f)
        return schemas

    def extract_json(self, response_content):
        json_match = re.search(r'\[.*\]|\{.*\}', response_content, re.DOTALL)
        if json_match:
            return json_match.group()
        return None

    def validate_response(self, response_content):
        json_content = self.extract_json(response_content)
        if not json_content:
            return False, "No JSON-like structure found in the response", None

        try:
            data = json.loads(json_content)
            
            # Determine the category based on the structure of the data
            category = self.determine_category(data)
            
            # Validate against the schema
            validate(instance=data, schema=self.schemas[category])
            
            return True, "Validation successful", data
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except ValidationError as e:
            return False, f"Schema validation failed: {e.message}", None
        except KeyError:
            return False, f"No schema found for the response structure", None
        except Exception as e:
            return False, f"Validation error: {str(e)}", None

    def determine_category(self, data):
        # Implement logic to determine the category based on the data structure
        # This is a placeholder implementation and should be adjusted based on your specific needs
        if isinstance(data, list):
            if all('Name' in item for item in data):
                return 'top_k'
        elif isinstance(data, dict):
            if 'twin1' in data and 'twin2' in data:
                return 'twins'
        # Add more conditions as needed
        raise ValueError("Unable to determine category from data structure")