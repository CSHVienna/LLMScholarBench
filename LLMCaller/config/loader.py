import json
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_category_variables():
    """Load category variables configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(script_dir, "category_variables.json")
    with open(config_path, 'r') as f:
        categories = json.load(f)
    
    return categories

def load_enhanced_criteria_description():
    """Load the criteria descriptions for all parameter types."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "criteria_description.json")
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

def load_domain_config(discipline=None):
    """Load domain-specific configuration from academic_disciplines.json with support for multiple disciplines."""
    academic_disciplines = load_academic_disciplines()
    
    # If discipline is not specified, raise an error to enforce explicit specification
    if not discipline:
        raise ValueError("Discipline must be explicitly specified. Use get_available_disciplines() to see options.")
    
    # Search through all categories for the specified discipline
    for category_name, category_data in academic_disciplines.items():
        disciplines = category_data.get("disciplines", {})
        if discipline in disciplines:
            discipline_data = disciplines[discipline]
            # Convert academic_disciplines format to domain_config format for backward compatibility
            return {
                "domain_name": discipline_data.get("name", discipline.replace("_", " ").title()),
                "domain_description": discipline_data.get("description", f"field of {discipline}"),
                "publication_source": ", ".join(discipline_data.get("publication_sources", ["academic journals"])[:3]) + " and other leading journals",
                "publication_description": f"prestigious journals such as {', '.join(discipline_data.get('publication_sources', ['academic journals'])[:2])}",
                "domain_expertise_title": f"expert research assistant responsible for compiling a list of leading {discipline_data.get('name', discipline.replace('_', ' '))} researchers",
                "domain_context": f"who have published in leading {discipline_data.get('name', discipline.replace('_', ' ')).lower()} journals and are members of professional societies such as {', '.join(discipline_data.get('major_societies', ['relevant professional societies'])[:2])}"
            }
    
    # Fallback for unknown disciplines
    return {
        "domain_name": discipline.replace("_", " ").title(),
        "domain_description": f"field of {discipline.replace('_', ' ')}",
        "publication_source": "academic journals",
        "publication_description": f"peer-reviewed {discipline.replace('_', ' ')} journals",
        "domain_expertise_title": f"expert research assistant responsible for compiling a list of leading {discipline.replace('_', ' ')} researchers",
        "domain_context": f"who have published in {discipline.replace('_', ' ')} journals"
    }

def get_available_disciplines():
    """Get list of all available disciplines across all categories."""
    academic_disciplines = load_academic_disciplines()
    
    disciplines = []
    for category_name, category_data in academic_disciplines.items():
        disciplines_in_category = category_data.get("disciplines", {})
        disciplines.extend(list(disciplines_in_category.keys()))
    
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