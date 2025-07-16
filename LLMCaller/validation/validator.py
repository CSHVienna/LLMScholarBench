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
        # First, try to extract JSON from markdown code blocks
        markdown_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', response_content, re.DOTALL)
        if markdown_match:
            return markdown_match.group(1)
        
        # Fallback to original extraction method
        json_match = re.search(r'\[.*\]|\{.*\}', response_content, re.DOTALL)
        if json_match:
            return json_match.group()
        return None

    def validate_response(self, response_content):
        # Clean up the response to handle common issues
        # Remove any reasoning explanation that might follow the JSON
        if "### Reasoning Explanation" in response_content:
            json_part = response_content.split("### Reasoning Explanation")[0].strip()
        elif "Reasoning Explanation" in response_content:
            json_part = response_content.split("Reasoning Explanation")[0].strip()
        else:
            json_part = response_content.strip()

        # Find the last closing brace or bracket to ensure we get all the JSON
        last_brace = json_part.rfind('}')
        last_bracket = json_part.rfind(']')

        end_pos = max(last_brace, last_bracket)
        if end_pos != -1:
            # If the response is a list, the end should be `]`
            if last_bracket > last_brace and json_part.strip().startswith('['):
                 json_part = json_part[:last_bracket + 1]
            # if the response is a dictionary
            elif last_brace > last_bracket and json_part.strip().startswith('{'):
                 json_part = json_part[:last_brace + 1]
            else:
                 json_part = json_part[:end_pos + 1]

        json_part = json_part.strip()
        if json_part.endswith("%"):
            json_part = json_part[:-1]

        try:
            # Attempt to parse the JSON from the cleaned response text
            data = json.loads(json_part)
            
            # The scholar search returns a dict with a 'scholars' key
            if isinstance(data, dict) and "scholars" in data:
                # We want the scholar object, not the wrapper
                if data["scholars"]:
                    return True, "Valid scholar search response.", data["scholars"][0]
                else:
                    return False, "Scholar search returned no scholars.", None

            # Check if the data is a list of dictionaries (expected format for most)
            if not isinstance(data, list):
                # Allow single dictionary for other cases too, just in case
                if isinstance(data, dict):
                    return True, "Valid single JSON object.", data
                return False, "Response is not a list or a valid dict.", None

            if not all(isinstance(item, dict) for item in data):
                return False, "Not all items in the list are dictionaries.", None

            return True, "Valid JSON list of objects.", data

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}", None

    def determine_category(self, data):
        # Implement logic to determine the category based on the data structure
        # This is a placeholder implementation and should be adjusted based on your specific needs
        if isinstance(data, list):
            if all('Name' in item and 'Relationship' in item for item in data):
                return 'genealogy'
            elif all('Name' in item for item in data):
                return 'top_k'
        elif isinstance(data, dict):
            if 'twin1' in data and 'twin2' in data:
                return 'twins'
        # Add more conditions as needed
        raise ValueError("Unable to determine category from data structure")