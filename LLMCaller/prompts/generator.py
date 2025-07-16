import json
import os
import sys
import argparse
from typing import Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

from config.loader import (
    load_twin_scientists_config, 
    load_definitions, 
    load_domain_config, 
    load_instructions_config,
    load_prompt_config,
    get_available_disciplines,
    get_discipline_info
)

def load_config(file_name: str) -> Dict[str, Any]:
    file_path = os.path.join(PARENT_DIR, "config", file_name)
    with open(file_path, 'r') as f:
        return json.load(f)

def load_template() -> str:
    template_path = os.path.join(SCRIPT_DIR, "prompt_template.txt")
    with open(template_path, 'r') as f:
        return f.read()

def generate_definitions_section(category: str, variable: str) -> str:
    """Generate the definitions section for the prompt based on category."""
    definitions = load_definitions()
    
    if category == 'twins':
        twins_def = definitions['statistical_twins']
        criteria_list = '\n'.join([f"- {criterion}" for criterion in twins_def['criteria']])
        return f"""### Definition ###

Statistical Twins: {twins_def['definition']}
{criteria_list}"""
    
    elif category == 'seniority':
        career_def = definitions['career_stages'][variable]
        criteria_list = '\n'.join([f"- {criterion}" for criterion in career_def['criteria']])
        return f"""### Definition ###

{career_def['definition']}
{criteria_list}"""
    
    return ""  # No definitions section for other categories

def generate_criteria(category: str, variable: str, criteria_description: Dict[str, Any], prompt_config: Dict[str, Any], twin_config: Dict[str, Any] = None, domain_config: Dict[str, Any] = None) -> str:
    cat_config = criteria_description[category]
    var_config = cat_config['variables'][variable]
    
    main_criteria = var_config['main_criteria']
    secondary_criteria = var_config['secondary_criteria']

    if category == 'twins' and twin_config:
        scientist_name = twin_config.get(variable, {}).get('name', 'Unknown Scientist')
        secondary_criteria = secondary_criteria.format(scientist_name=scientist_name)

    criteria_phrase = cat_config['criteria_phrase']
    
    # Prepare format parameters
    format_params = {
        'main_criteria': main_criteria,
        'secondary_criteria': secondary_criteria,
        'affiliation_context': '' # Default to empty string
    }

    if category == 'scholar_search':
        # Load affiliation from runtime_config.json if it exists
        try:
            config_path = os.path.join(PARENT_DIR, "config", "runtime_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    runtime_config = json.load(f)
                    affiliation = runtime_config.get('affiliation')
                    if affiliation:
                        format_params['affiliation_context'] = f" from {affiliation}"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Handle cases where the file doesn't exist or is empty/corrupt
            pass
    
    # Handle genealogy scholar_name placeholder
    if category == 'genealogy':
        # Get scholar name from environment variable or default
        scholar_name = os.environ.get('SCHOLAR_NAME', 'Dr. Albert Einstein')
        format_params['scholar_name'] = scholar_name
    
    # Add domain-specific parameters if available
    if domain_config:
        format_params['domain_context'] = domain_config.get('domain_context', '')
        format_params['publication_description'] = domain_config.get('publication_description', '')
    
    return criteria_phrase.format(**format_params)

def format_step_instructions(steps: list, criteria: str, backup_indicator: str) -> str:
    """Format the step instructions as a numbered list."""
    formatted_steps = []
    for i, step in enumerate(steps, 1):
        formatted_step = step.format(criteria=criteria, backup_indicator=backup_indicator)
        formatted_steps.append(f"{i}. {formatted_step}")
    return '\n'.join(formatted_steps)

def format_additional_guidelines(guidelines: list) -> str:
    """Format additional guidelines as bullet points."""
    return '\n'.join([f"- {guideline}" for guideline in guidelines])

def generate_prompt(category: str, variable: str, discipline: str) -> str:
    criteria_description = load_config("criteria_description.json")
    prompt_config = load_prompt_config()
    template = load_template()
    twin_config = load_twin_scientists_config() if category == 'twins' else None
    
    if discipline == 'all':
        domain_config = {
            "domain_expertise_title": "expert research assistant",
            "domain_description": "any academic field",
            "domain_context": "",
            "publication_description": "academic journals"
        }
    else:
        domain_config = load_domain_config(discipline)

    instructions_config = load_instructions_config()

    criteria = generate_criteria(category, variable, criteria_description, prompt_config, twin_config, domain_config)
    definitions_section = generate_definitions_section(category, variable)
    
    cat_prompt_config = prompt_config[category]

    backup_indicator = cat_prompt_config['backup_indicator']
    # Customizing the backup_indicator for 'field'
    if category == 'field':
        display = criteria_description[category]['variables'][variable]['display']
        backup_indicator = cat_prompt_config['backup_indicator'].format(display=display)

    output_example = cat_prompt_config['output_example']

    # Customizing the output_example for 'top_k'
    if category == 'top_k':
        max_scientists = int(variable.split('_')[-1])
        if max_scientists <= 5:
            output_example = ', '.join([f'{{"Name": "Scientist {i}"}}' for i in range(1, max_scientists + 1)])
            output_example = f'[{output_example}]'
        else:
            output_example = ', '.join([f'{{"Name": "Scientist {i}"}}' for i in range(1, 4)])  # Show first 3 scientists
            output_example += ', ..., ' + f'{{"Name": "Scientist {max_scientists}"}}'
            output_example = f'[{output_example}]'

    # Special handling for scholar_search
    if category == 'scholar_search':
        # Check if scholar_name is in config (set in main.py)
        import os
        config_path = os.path.join(PARENT_DIR, "config", "runtime_config.json")
        scholar_name = ""
        
        # Try to load the scholar name from runtime config if available
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    runtime_config = json.load(f)
                    scholar_name = runtime_config.get('scholar_name', '')
        except Exception as e:
            print(f"Warning: Could not load scholar name from runtime config: {e}")
        
        # If no scholar name in runtime config, check environment variable
        if not scholar_name:
            scholar_name = os.environ.get('SCHOLAR_NAME', 'Dr. Archit Somani')  # Default to the test scholar if not specified
        
        # Replace the placeholder in the criteria
        criteria = criteria.replace('{scholar_name}', scholar_name)
    
    # Special handling for genealogy
    if category == 'genealogy':
        # Check if scholar_name is in config (set in main.py)
        import os
        config_path = os.path.join(PARENT_DIR, "config", "runtime_config.json")
        scholar_name = ""
        
        # Try to load the scholar name from runtime config if available
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    runtime_config = json.load(f)
                    scholar_name = runtime_config.get('scholar_name', '')
        except Exception as e:
            print(f"Warning: Could not load scholar name from runtime config: {e}")
        
        # If no scholar name in runtime config, check environment variable
        if not scholar_name:
            scholar_name = os.environ.get('SCHOLAR_NAME', 'Dr. Albert Einstein')  # Default to Einstein for genealogy testing
        
        # Replace the placeholder in the criteria
        criteria = criteria.replace('{scholar_name}', scholar_name)

    # Format step instructions
    step_instructions = format_step_instructions(
        instructions_config['common_steps'], 
        criteria, 
        backup_indicator
    )
    
    # Format additional guidelines
    additional_guidelines = format_additional_guidelines(
        instructions_config['additional_guidelines']
    )
    
    # Format task instruction
    task_instruction = instructions_config['task_instruction_template'].format(criteria=criteria)

    prompt_data = {
        'domain_expertise_title': domain_config['domain_expertise_title'],
        'domain_description': domain_config['domain_description'],
        'domain_context': domain_config['domain_context'],
        'criteria': criteria,
        'definitions_section': definitions_section,
        'task_instruction': task_instruction,
        'step_instructions': step_instructions,
        'output_format_instruction': instructions_config['output_format_instruction'],
        'output_example': output_example,
        'additional_guidelines': additional_guidelines,
        'reasoning_instruction': instructions_config['reasoning_instruction']
    }
    
    return template.format(**prompt_data)

def save_prompts_to_file(prompts: Dict[str, Dict[str, str]], discipline: str) -> None:
    output_file = os.path.join(SCRIPT_DIR, f"final_prompts_{discipline}.txt")
    with open(output_file, 'w') as f:
        f.write(f"# Academic Prompts for {discipline.title()}\n")
        f.write(f"# Generated using multi-discipline configuration system\n\n")
        
        for category, variables in prompts.items():
            f.write(f"{'=' * 50}\n")
            f.write(f"Category: {category.upper()}\n")
            f.write(f"{'=' * 50}\n\n")
            for variable, prompt in variables.items():
                f.write(f"--- Variable: {variable} ---\n\n")
                f.write(prompt)
                f.write("\n\n")
    print(f"Prompts for {discipline} have been saved to {output_file}")

def list_available_disciplines():
    """List all available disciplines."""
    disciplines = get_available_disciplines()
    print("Available disciplines:")
    for discipline in disciplines:
        info = get_discipline_info(discipline)
        if info:
            print(f"  - {discipline} ({info['category_name']})")
        else:
            print(f"  - {discipline}")

def generate_for_discipline(discipline: str) -> None:
    """Generate prompts for a specific discipline."""
    criteria_description = load_config("criteria_description.json")
    generated_prompts = {}

    print(f"Generating prompts for {discipline}...")
    
    for category, config in criteria_description.items():
        generated_prompts[category] = {}
        for variable in config['variables']:
            prompt = generate_prompt(category, variable, discipline)
            generated_prompts[category][variable] = prompt
            print(f"Generated prompt for {category}: {variable} (discipline: {discipline})")

    save_prompts_to_file(generated_prompts, discipline)

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate academic prompts for various disciplines')
    parser.add_argument('--discipline', '-d', type=str,
                        help='Academic discipline to generate prompts for')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available disciplines')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate prompts for all available disciplines')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_disciplines()
        return
    
    if args.all:
        disciplines = get_available_disciplines()
        print(f"Generating prompts for all {len(disciplines)} disciplines...")
        for discipline in disciplines:
            generate_for_discipline(discipline)
        print("All disciplines completed!")
    elif args.discipline:
        generate_for_discipline(args.discipline)
    else:
        print("Error: Please specify a discipline with --discipline, use --list to see options, or use --all to generate for all disciplines.")
        list_available_disciplines()

if __name__ == "__main__":
    main()