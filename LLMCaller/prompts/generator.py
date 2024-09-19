import json
import os
import sys
from typing import Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

from config.loader import load_twin_scientists_config

def load_config(file_name: str) -> Dict[str, Any]:
    file_path = os.path.join(PARENT_DIR, "config", file_name)
    with open(file_path, 'r') as f:
        config_data = f.read().replace("'", '"')  # Replace single quotes with double quotes
        return json.loads(config_data)

def load_template() -> str:
    template_path = os.path.join(SCRIPT_DIR, "prompt_template.txt")
    with open(template_path, 'r') as f:
        return f.read()

def generate_criteria(category: str, variable: str, criteria_description: Dict[str, Any], prompt_config: Dict[str, Any], twin_config: Dict[str, Any] = None) -> str:
    cat_config = criteria_description[category]
    var_config = cat_config['variables'][variable]
    
    main_criteria = var_config['main_criteria']
    secondary_criteria = var_config['secondary_criteria']

    if category == 'twins' and twin_config:
        scientist_name = twin_config.get(variable, {}).get('name', 'Unknown Scientist')
        secondary_criteria = secondary_criteria.format(scientist_name=scientist_name)

    criteria_phrase = cat_config['criteria_phrase']
    return criteria_phrase.format(main_criteria=main_criteria, secondary_criteria=secondary_criteria)

def generate_prompt(category: str, variable: str) -> str:
    criteria_description = load_config("criteria_description.json")
    prompt_config = load_config("prompt_config.json")
    template = load_template()
    twin_config = load_twin_scientists_config() if category == 'twins' else None

    criteria = generate_criteria(category, variable, criteria_description, prompt_config, twin_config)
    
    cat_prompt_config = prompt_config[category]

    backup_indicator = cat_prompt_config['backup_indicator']
    # Customizing the output_example for 'field'
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

    prompt_data = {
        'criteria': criteria,
        'backup_indicator': backup_indicator,
        'output_example': output_example
    }
    
    return template.format(**prompt_data)

def save_prompts_to_file(prompts: Dict[str, Dict[str, str]]) -> None:
    output_file = os.path.join(SCRIPT_DIR, "final_prompts.txt")
    with open(output_file, 'w') as f:
        for category, variables in prompts.items():
            f.write(f"{'=' * 50}\n")
            f.write(f"Category: {category.upper()}\n")
            f.write(f"{'=' * 50}\n\n")
            for variable, prompt in variables.items():
                f.write(f"--- Variable: {variable} ---\n\n")
                f.write(prompt)
                f.write("\n\n")
    print(f"Prompts have been saved to {output_file}")

def main() -> None:
    criteria_description = load_config("criteria_description.json")
    generated_prompts = {}

    for category, config in criteria_description.items():
        generated_prompts[category] = {}
        for variable in config['variables']:
            prompt = generate_prompt(category, variable)
            generated_prompts[category][variable] = prompt
            print(f"Generated prompt for {category}: {variable}")

    save_prompts_to_file(generated_prompts)

if __name__ == "__main__":
    main()