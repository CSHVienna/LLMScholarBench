import json
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_category_variables():
    """Load the category variables configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "category_variables.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_llm_setup(model_name):
    """Load the LLM setup configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "llm_setup.json")
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    return all_configs.get(model_name, {})

def load_twin_scientists_config():
    """Load the twin scientists configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "twin_scientists_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_definitions():
    """Load the definitions configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "definitions.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_academic_disciplines():
    """Load the comprehensive academic disciplines configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "academic_disciplines.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_domain_config(discipline="physics"):
    """Load domain-specific configuration with support for multiple disciplines."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "domain_config.json")
    with open(config_path, 'r') as f:
        domain_config = json.load(f)
    
    # If discipline is not specified, use default
    if not discipline:
        discipline = domain_config.get("default_discipline", "physics")
    
    # Search through all categories for the specified discipline
    discipline_categories = domain_config.get("discipline_categories", {})
    
    for category_name, disciplines in discipline_categories.items():
        if discipline in disciplines:
            return disciplines[discipline]
    
    # Fallback to physics if discipline not found
    for category_name, disciplines in discipline_categories.items():
        if "physics" in disciplines:
            return disciplines["physics"]
    
    # Ultimate fallback
    return {
        "domain_name": discipline,
        "domain_description": f"field of {discipline}",
        "publication_source": "academic journals",
        "publication_description": f"peer-reviewed {discipline} journals",
        "domain_expertise_title": f"expert research assistant responsible for compiling a list of leading {discipline} researchers",
        "domain_context": f"who have published in {discipline} journals"
    }

def get_available_disciplines():
    """Get list of all available disciplines across all categories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "domain_config.json")
    with open(config_path, 'r') as f:
        domain_config = json.load(f)
    
    disciplines = []
    discipline_categories = domain_config.get("discipline_categories", {})
    
    for category_name, category_disciplines in discipline_categories.items():
        disciplines.extend(list(category_disciplines.keys()))
    
    return sorted(disciplines)

def get_discipline_info(discipline):
    """Get comprehensive information about a specific discipline."""
    academic_disciplines = load_academic_disciplines()
    
    for category_name, category_data in academic_disciplines.items():
        disciplines = category_data.get("disciplines", {})
        if discipline in disciplines:
            return {
                "category": category_name,
                "category_name": category_data.get("category_name"),
                "discipline_info": disciplines[discipline]
            }
    
    return None

def load_instructions_config():
    """Load common instructions configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "instructions_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_validation_config():
    """Load validation rules configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "validation_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def load_prompt_config():
    """Load prompt-specific configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "prompt_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)