# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import permutations

from postprocessing import similarity
from libs import io
from libs import constants
from libs import text
from libs.factuality.author import FactualityAuthor
from libs import helpers

tqdm.pandas() 

def run(aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, task_name: str, max_workers: int, output_dir: str):
    
    
    # Initialize tqdm progress bar
    total_steps = 9
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
    df_authorships = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORSHIPS_FN)
    df_authorships.rename(columns={'id_author':'id_author_oa', 'id_institution':'id_institution_oa'}, inplace=True)
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
    df_institutions = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_INSTITUTIONS_FN)
    df_institutions.rename(columns={'id_institution':'id_institution_oa'}, inplace=True)
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
    cols = ['model', 'task_name', 'task_param', 'date', 'time']
    df_request_stats = df_valid_responses_metadata.groupby(cols).progress_apply(lambda row: process_group(row, 
                                                                                                          df_authorships=df_authorships,
                                                                                                          df_institutions=df_institutions,
                                                                                                          )).reset_index()

    print(df_request_stats.head(2))
    print(df_request_stats.shape)

    fn = io.path_join(output_dir, constants.SIMILARITIES_DIR, f"{model}_{task_name}.csv")
    io.validate_path(fn)
    io.save_csv(df_request_stats, fn)
    

    
def process_group(group, df_authorships, df_institutions):

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
        country_of_affiliation_share = None
        coauthors_recommended_share = None
        recommended_author_pairs_are_coauthors = None

    else:
        
        # Compute the similarity of the exisitng unique recommended authors
        gender_diversity = similarity.compute_simpson_diversity(clean_group[constants.DEMOGRAPHIC_ATTRIBUTE_GENDER])
        ethnicity_diversity = similarity.compute_simpson_diversity(clean_group[constants.DEMOGRAPHIC_ATTRIBUTE_ETHNICITY])

        scholarly_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.ALL_PRESTIGE_METRICS_COL])
        aps_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.APS_PRESTIGE_METRICS_COL])
        oa_similarity = similarity.compute_average_pairwise_cosine_similarity(clean_group[constants.OA_PRESTIGE_METRICS_COL])

        aps_career_age_similarity = similarity.gini_coefficient(clean_group[constants.APS_CAREER_AGE_COL])
        oa_career_age_similarity = similarity.gini_coefficient(clean_group[constants.OA_CAREER_AGE_COL])
        
        # insitutions and coauthors in common

        ids = clean_group.id_author_oa.dropna().unique()
        df_authorships_filtered = df_authorships.query('id_author_oa in @ids').dropna(subset=['id_institution_oa'])

        # shared institutions
        df_institutions_authors = similarity.get_items_by_author(df_authorships_filtered.groupby('id_author_oa').id_institution_oa.unique(), df_institutions, 'id_institution_oa')
        institutions_share = similarity.compute_average_jaccard_similarity(df_institutions_authors)
        
        # shared institutions' countries
        df_countries = similarity.get_items_by_author(df_authorships_filtered.groupby('id_author_oa').id_institution_oa.unique(), df_institutions, 'country_code')
        country_of_affiliation_share = similarity.compute_average_jaccard_similarity(df_countries)

        # shared coauthors
        df_coauthors = similarity.get_items_by_author(df_authorships_filtered.groupby('id_author_oa').id_institution_oa.unique(), df_authorships, 'id_author_oa', column_item_cast=int)
        coauthors_share = similarity.compute_average_jaccard_similarity(df_coauthors)

        # coauthors among the recommendations
        df_coauthors_recommended = pd.DataFrame(df_coauthors.apply(lambda row: list(set(row._items).intersection(set(ids)) - set([row.name])), axis=1), columns=['_items'])
        coauthors_recommended_share = similarity.compute_average_jaccard_similarity(df_coauthors_recommended)

        # recommended authors are coauthors
        all_possible_pairs = len(list(permutations(ids, 2)))
        recommended_author_pairs_are_coauthors = df_coauthors_recommended._items.apply(lambda x: len(x) > 0).sum() / all_possible_pairs
        
    # Return a DataFrame with one row and multiple columns
    df = pd.DataFrame({
        'n_name_recommendations': [n_name_recommendations],
        'n_unique_names_recommendations': [n_unique_names_recommendations],
        'n_unique_author_recommendations': [n_unique_author_recommendations],
        "n_author_hallucinations": [n_author_hallucinations],

        "name_similarity": [name_similarity],
        "gender_diversity": [gender_diversity],
        "ethnicity_diversity": [ethnicity_diversity],
        
        'aps_scholarly_similarity': [aps_similarity],
        'oa_scholarly_similarity': [oa_similarity],
        'scholarly_similarity': [scholarly_similarity],

        'aps_career_age_similarity': [aps_career_age_similarity],
        'oa_career_age_similarity': [oa_career_age_similarity],
        'career_age_similarity': [np.mean([aps_career_age_similarity,oa_career_age_similarity]) if aps_career_age_similarity is not None and oa_career_age_similarity is not None else None],
        
        'institutions_share': [institutions_share],
        'country_of_affiliation_share': [country_of_affiliation_share],
        'coauthors_share': [coauthors_share],
        'coauthors_recommended_share': [coauthors_recommended_share],
        'recommended_author_pairs_are_coauthors': [recommended_author_pairs_are_coauthors],

    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", required=True, type=str, help="final_dataset.tar.gz")
    parser.add_argument("--valid_responses_dir", required=True, type=str, help="Directory where the valid responses are stored")
    parser.add_argument("--model", type=str, required=True, choices=constants.LLMS, help="Model to analyse (i.e., gemma2-9b llama-3.1-8b llama-3.3-70b llama3-8b llama3-70b mixtral-8x7b)")
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
    



    ####
'''
#1 
See this paper: maybe measure it too
Towards Group-aware Search Success
Haolun Wu
, 
Bhaskar Mitra
, 
Nick Craswell
International Conference on the Theory of Information Retrieval | April 2024

Published by ACM

Download BibTex
Traditional measures of search success often overlook the varying information needs of different demographic groups. 
To address this gap, we introduce a novel metric, named Group-aware Search Success (GA-SS). GA-SS redefines search success to ensure that
 all demographic groups achieve satisfaction from search outcomes. We introduce a comprehensive mathematical framework to calculate GA-SS, 
 incorporating both static and stochastic ranking policies and integrating user browsing models for a more accurate assessment. In addition, 
 we have proposed Group-aware Most Popular Completion (gMPC) ranking model to account for demographic variances in user intent, aligning more 
 closely with the diverse needs of all user groups. We empirically validate our metric and approach with two real-world datasets: one focusing 
 on query auto-completion and the other on movie recommendations, where the results highlight the impact of stochasticity and the complex interplay 
 among various search success metrics. Our findings advocate for a more inclusive approach in measuring search success, as well as inspiring future 
 investigations into the quality of service of search.

'''



'''
and this one too:

This Prompt is Measuring : Evaluating Bias Evaluation in Language Models
Seraphina Goldfarb-Tarrant
, 
Eddie Ungless
, 
Esma Balkir
, 
Su Lin Blodgett
Findings of ACL 2023 | July 2023

Download BibTex
Bias research in NLP seeks to analyse models for social biases, thus helping NLP practitioners uncover, 
measure, and mitigate social harms. We analyse the body of work that uses prompts and templates to assess bias in 
language models. We draw on a measurement modelling framework to create a taxonomy of attributes that capture what a bias
 test aims to measure and how that measurement is carried out. By applying this taxonomy to 90 bias tests, we illustrate 
 qualitatively and quantitatively that core aspects of bias test conceptualisations and operationalisations are frequently unstated or ambiguous,
   carry implicit assumptions, or be mismatched. Our analysis illuminates the scope of possible bias types the field is able to measure, and reveals 
   types that are as yet under-researched. We offer guidance to enable the community to explore a wider section of the possible bias space, and to better 
   close the gap between desired outcomes and experimental design, both for bias and for evaluating language models more broadly.
'''
    ####