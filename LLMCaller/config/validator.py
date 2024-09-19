from jsonschema import validate

def validate_llm_setup(config):
    schema = {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "max_attempts": {"type": "integer", "minimum": 1},
            "temperature": {"type": "number", "minimum": 0, "maximum": 1},
            "max_tokens": {"type": "integer", "minimum": 1},
            "top_p": {"type": "number", "minimum": 0, "maximum": 1},
            "stop": {
                "type": ["array", "null"],
                "items": {"type": "string"}
            },
            "stream": {"type": "boolean"},
            "system_message": {"type": "string"}
        },
        "required": ["model", "max_attempts", "temperature", "max_tokens", "top_p", "system_message", "stream"]
    }
    validate(instance=config, schema=schema)



def validate_category_variables(config):
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
    validate(instance=config, schema=schema)

def validate_twin_scientists_config(config):
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"}
    }
    validate(instance=config, schema=schema)