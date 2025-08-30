import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load the JSON file
file_path = "././experiments_validation_results/updated_organized_results/descriptive_statistics_by_run.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to compute avg and std for the ratio
def compute_ratio_statistics(count_names, count_other):
    ratios = []
    for name, other in zip(count_names, count_other):
        if name != 0:  # Ensure total names > 0
            ratios.append(other / name)  # Keep the 0's in the other counts (OA or APS)
    
    if not ratios:
        return '--', '--'
    
    avg = np.mean(ratios)
    std = np.std(ratios)
    return f'{avg:.2f}', f'{std:.2f}'

# Function to process factuality ratios for the 'field', 'epoch', and 'seniority' use cases
def process_model_factuality(config_data):
    overall_ratio_oa_list = []
    overall_ratio_aps_list = []
    overall_ratio_author_correct_field_list = []
    overall_ratio_paper_correct_field_list = []
    overall_ratio_author_correct_epoch_list = []
    overall_ratio_author_correct_seniority_list = []
    overall_career_age_error_list = []  # For the mean career age error

    for use_case, metrics in config_data.items():
        mask = [name != 0 for name in metrics['count_names']]
        overall_ratio_oa_list.extend([oa / name for name, oa, m in zip(metrics['count_names'], metrics['count_in_oa'], mask) if m])
        overall_ratio_aps_list.extend([aps / name for name, aps, m in zip(metrics['count_names'], metrics['count_in_aps'], mask) if m])
        
        # Process 'field' use cases
        if use_case.startswith('field'):
            
            
            overall_ratio_author_correct_field_list.extend([acf / name for name, acf, m in zip(metrics['count_names'], metrics.get('author_correct_field', []), mask) if m])
            
            if 'count_correct_field' in metrics:
                overall_ratio_paper_correct_field_list.extend([cf / name for name, cf, m in zip(metrics['count_names'], metrics['count_correct_field'], mask) if m])

        # Process 'epoch' use cases
        elif use_case.startswith('epoch'):
            mask = [name != 0 for name in metrics['count_names']]
            overall_ratio_author_correct_epoch_list.extend([cae / name for name, cae, m in zip(metrics['count_names'], metrics.get('count_correct_author_epoch', []), mask) if m])

        # Process 'seniority' use cases
        elif use_case.startswith('seniority'):
            mask = [name != 0 for name in metrics['count_names']]
            overall_ratio_author_correct_seniority_list.extend([cas / name for name, cas, m in zip(metrics['count_names'], metrics.get('count_correct_author_seniority', []), mask) if m])
            
            # Check if 'mean_career_age_error' exists and append the list if it does
            if 'mean_career_age_error' in metrics:
                overall_career_age_error_list.extend([error for error, m in zip(metrics['mean_career_age_error'], mask) if m])

    # Compute avg and std for OA, APS, and seniority ratios
    factuality_oa_avg, factuality_oa_std = compute_ratio_statistics([1] * len(overall_ratio_oa_list), overall_ratio_oa_list)
    factuality_aps_avg, factuality_aps_std = compute_ratio_statistics([1] * len(overall_ratio_aps_list), overall_ratio_aps_list)
    factuality_author_correct_field_avg, factuality_author_correct_field_std = compute_ratio_statistics([1] * len(overall_ratio_author_correct_field_list), overall_ratio_author_correct_field_list)
    factuality_paper_correct_field_avg, factuality_paper_correct_field_std = compute_ratio_statistics([1] * len(overall_ratio_paper_correct_field_list), overall_ratio_paper_correct_field_list)

    # For epoch use cases
    factuality_author_correct_epoch_avg, factuality_author_correct_epoch_std = compute_ratio_statistics([1] * len(overall_ratio_author_correct_epoch_list), overall_ratio_author_correct_epoch_list)

    # For seniority use cases
    factuality_author_correct_seniority_avg, factuality_author_correct_seniority_std = compute_ratio_statistics([1] * len(overall_ratio_author_correct_seniority_list), overall_ratio_author_correct_seniority_list)

    # Career age error (mean ± std)
    overall_career_age_error_list = [e for e in overall_career_age_error_list if pd.notna(e) and e is not None]
    if overall_career_age_error_list:
        career_age_error_avg = np.mean(overall_career_age_error_list)
        career_age_error_std = np.std(overall_career_age_error_list)
        career_age_error = f"{career_age_error_avg:.2f} ± {career_age_error_std:.2f}"
    else:
        career_age_error = "--"

    return (factuality_oa_avg + " ± " + factuality_oa_std,
            factuality_aps_avg + " ± " + factuality_aps_std,
            factuality_author_correct_field_avg + " ± " + factuality_author_correct_field_std,
            factuality_paper_correct_field_avg + " ± " + factuality_paper_correct_field_std,
            factuality_author_correct_epoch_avg + " ± " + factuality_author_correct_epoch_std,
            factuality_author_correct_seniority_avg + " ± " + factuality_author_correct_seniority_std,
            career_age_error)

# Function to create the summary table
def create_factuality_summary(data, models):
    #models = ['llama3-8b', 'gemma2-9b', 'mixtral-8x7b', 'llama3-70b']
    factuality_summary = {
        'Factuality OA': [],
        'Factuality APS': [],
        'Author Correct Field Ratio': [],
        'Paper Correct Field Ratio': [],
        'Author Correct Epoch': [],
        'Correct Seniority Attribution': [],  # Adding seniority attribution row
        'Mean Error from Actual Career Age': []  # Adding career age error row
    }
    
    for model in models:
        for config_name in data.keys():
            if model in config_name:
                factuality_oa, factuality_aps, factuality_author_correct, factuality_paper_correct, factuality_author_epoch, factuality_author_seniority, career_age_error = process_model_factuality(data[config_name])
                factuality_summary['Factuality OA'].append(factuality_oa)
                factuality_summary['Factuality APS'].append(factuality_aps)
                factuality_summary['Author Correct Field Ratio'].append(factuality_author_correct)
                factuality_summary['Paper Correct Field Ratio'].append(factuality_paper_correct)
                factuality_summary['Author Correct Epoch'].append(factuality_author_epoch)
                factuality_summary['Correct Seniority Attribution'].append(factuality_author_seniority)
                factuality_summary['Mean Error from Actual Career Age'].append(career_age_error)
                break  # Break after finding the config for this model
    
    # Create a DataFrame from the factuality summary
    factuality_df = pd.DataFrame(factuality_summary, index=models)
    factuality_df.index.name = 'Model'
    
    return factuality_df.transpose()

# Function to save the summary table as PNG and TEX
def save_factuality_summary(df, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save as PNG
    plt.figure(figsize=(8, 3))
    plt.axis('off')
    plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    plt.tight_layout()
    png_path = os.path.join(folder_name, 'factuality_summary.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save as TEX
    tex_path = os.path.join(folder_name, 'factuality_summary.tex')
    with open(tex_path, 'w') as tex_file:
        tex_file.write("\\begin{table}[h]\n")
        tex_file.write("\\centering\n")
        tex_file.write("\\caption{Factuality Summary for OA, APS, Field, Epoch, and Seniority Ratios}\n")
        tex_file.write("\\label{tab:factuality_summary}\n")
        tex_file.write(df.to_latex(index=True))
        tex_file.write("\\end{table}\n")

# Main logic to create and save the factuality summary
if __name__ == "__main__":

    # Load model names from the config file
    config_file = os.path.join('../model_config.json')
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        models = config_data["models"]

    # Create the factuality summary table
    factuality_summary_df = create_factuality_summary(data, models)
    base_output_folder = 'output_results/factuality'
    output_folder = os.path.join(base_output_folder, 'factuality_summary_table')
    os.makedirs(output_folder, exist_ok=True)

    # Save the factuality summary as PNG and TEX
    save_factuality_summary(factuality_summary_df, output_folder)

    print(f"Factuality summary saved in folder: {output_folder}")

   
