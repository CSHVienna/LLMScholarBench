# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from postprocessing import similarity
from libs import io
from libs import constants
from libs import text
from libs.factuality.author import FactualityAuthor

tqdm.pandas() 

def run(aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, task_name: str, max_workers: int, output_dir: str):
    
    
    # Initialize tqdm progress bar
    total_steps = 10
    progress_bar = tqdm(total=total_steps, desc="Loading data")

    ### LLM valid answers

    fact_author = FactualityAuthor(aps_os_data_tar_gz=aps_os_data_tar_gz, valid_responses_dir=valid_responses_dir, model=model, max_workers=max_workers, output_dir=output_dir)
    if fact_author.already_checked():
        fact_author.load_valid_responses_with_factuality_check()
        progress_bar.update(1)

        req_cols = ['model', 'task_name', 'task_param', 'date', 'time', 'id_author_oa', 'clean_name']
        df_valid_responses = fact_author.df_valid_responses[req_cols].query("task_name == @task_name").copy() # filtering by task_name

        # We remove repeated names from the same request to avoid biasing the results
        df_valid_responses = df_valid_responses.drop_duplicates(subset=req_cols).copy()
        progress_bar.update(1)
        
    else:
        raise ValueError("Factuality check not found")
    
    if df_valid_responses.empty:
        io.printf(f"No valid responses found for model {model} and task_name {task_name}")
        return
    

    ### APS

    df_aps_stats = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHOR_STATS_FN)
    df_aps_stats.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    progress_bar.update(1)

    ### APS OA data

    # coathorship network: coathors per paper (id_publication, id_author, id_institution)
    # df_authorships = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORSHIPS_FN)
    # df_authorships.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    progress_bar.update(1)
    
    # demographics: gender, ethnicity
    df_author_demographics = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_DEMOGRAPHICS_FN)
    df_author_demographics.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    progress_bar.update(1)

    # scholarly metrics: two_year_mean_citedness, h_index, i10_index, works_count, cited_by_count
    df_author_stats = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_STATS_FN)
    df_author_stats.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    progress_bar.update(1)

    # institutions: id_institution, cited_by_count, country_code, 2yr_mean_citedness, h_index, i10_index, works_count, city, type
    # df_institutions = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_INSTITUTIONS_FN)
    progress_bar.update(1)

    # year of affiliation: id_author, id_institution, year
    # df_author_institution_years = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_INSTITUTION_YEAR_FN)
    # df_author_institution_years.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    progress_bar.update(1)


    
    

    ### MERGING

    df_author_stats_all = df_author_stats.merge(df_author_demographics, on='id_author_oa', how='left') # merging demographics with scholarly stats
    df_author_stats_all = df_author_stats_all.merge(df_aps_stats, on='id_author_oa', how='left')       # merging with APS stats
    progress_bar.update(1)

    df_valid_responses_metadata = df_valid_responses.merge(df_author_stats_all, on='id_author_oa', how='left')   # merging with llm responses
    progress_bar.update(1)


    ### STATS
    # 1. for each request (model, task_name, task_param, date, time) compute similarity of authors
    # 1.1 categorical (demographics + ...): gender, ethnicity, country of affiliations, affiliations, nobel prize winners, coauthors
    # 1.2 numerical (stats): works_count,cited_by_count,two_year_mean_citedness,h_index,i10_index,e_index,career_age,citations_per_paper_age
    # Output: for every request, similarity of authors (cosine similarity) and stats (mean, std, min, max, median, 25%, 75%)

    # stats (scholarly and demographics)
    df_request_stats = df_valid_responses_metadata.groupby(['model', 'task_name', 'task_param', 'date', 'time']).progress_apply(process_group).reset_index()

    print(df_request_stats.head(2))
    print(df_request_stats.shape)

    fn = io.path_join(output_dir, constants.SIMILARITIES_DIR, f"{model}_{task_name}.csv")
    io.validate_path(fn)
    io.save_csv(df_request_stats, fn)
    

    
def process_group(group):

    # Remove rows with missing author ids
    clean_group = group.dropna(subset=['id_author_oa'])
    
    # Compute the number of name recommendations and author hallucinations
    n_name_recommendations = group.shape[0]
    n_unique_names_recommendations = group.clean_name.nunique()
    n_unique_author_recommendations = clean_group.id_author_oa.nunique()
    n_author_hallucinations = n_unique_names_recommendations - n_unique_author_recommendations

    # Compute the similarity of all retrieved names (regardless of factuality)
    name_similarity = text.compute_similarity_list_of_text(group['clean_name'].drop_duplicates().values)

    if clean_group.empty or n_unique_author_recommendations == 1:
        gender_diversity = None
        ethnicity_diversity = None
        scholarly_similarity = None
        aps_similarity = None
        oa_similarity = None
        aps_career_age_similarity = None
        oa_career_age_similarity = None
        institutions_share = None
        coauthors_share = None
    
    else:
        
        # Compute the similarity of the exisitng unique recommended authors
        gender_diversity = similarity.compute_simpson_diversity(clean_group[constants.DEMOGRAPHIC_ATTRIBUTE_GENDER])
        ethnicity_diversity = similarity.compute_simpson_diversity(clean_group[constants.DEMOGRAPHIC_ATTRIBUTE_ETHNICITY])
        scholarly_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.ALL_SCHOLARLY_METRICS_COL])
        aps_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.APS_SCHOLARLY_METRICS_COL])
        oa_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.OA_SCHOLARLY_METRICS_COL])
        aps_career_age_similarity = similarity.compute_average_cosine_zscore_similarity(clean_group[constants.APS_CAREER_AGE_COL])
        oa_career_age_similarity = similarity.compute_average_cosine_zscore_similarity(clean_group[constants.OA_CAREER_AGE_COL])
        institutions_share = None #similarity.compute_average_cosine_similarity(clean_group[constants.CAREER_AGE_COL])
        coauthors_share = None #similarity.compute_average_cosine_similarity(clean_group[constants.CAREER_AGE_COL])

    # Return a DataFrame with one row and multiple columns
    df = pd.DataFrame({
        'n_name_recommendations': [n_name_recommendations],
        'n_unique_names_recommendations': [n_unique_names_recommendations],
        'n_unique_author_recommendations': [n_unique_author_recommendations],
        "n_author_hallucinations": [n_author_hallucinations],
        "name_similarity": [name_similarity],
        "gender_diversity": [gender_diversity],
        "ethnicity_diversity": [ethnicity_diversity],
        'scholarly_similarity': [scholarly_similarity],
        'aps_similarity': [aps_similarity],
        'oa_similarity': [oa_similarity],
        'aps_career_age_similarity': [aps_career_age_similarity],
        'oa_career_age_similarity': [oa_career_age_similarity],
        'institutions_share': [institutions_share],
        'coauthors_share': [coauthors_share]
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", required=True, type=str, help="final_dataset.tar.gz")
    parser.add_argument("--valid_responses_dir", required=True, type=str, help="Directory where the valid responses are stored")
    parser.add_argument("--model", type=str, required=True, choices=constants.LLMS, help="Model to analyse (i.e., gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b)")
    parser.add_argument("--task_name", type=str, choices=constants.EXPERIMENT_TASKS, help="Tasks to analyse (i.e., top_k field epoch seniority twins)")
    parser.add_argument("--max_workers", type=int, default=1, help="How many jobs to run in parallel maximum")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.valid_responses_dir, args.model, args.task_name, args.max_workers, args.output_dir)
    io.printf("Done!")
    