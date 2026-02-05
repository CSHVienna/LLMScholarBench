# Steps
# 1. read summaries
# 2. read valid responses
# 3. read factuality
# 4. read similarities
# 5. summarize metrics of interest per response (some will need summaries, others factuality)

# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from tqdm import tqdm
import pandas as pd

from libs import io
from libs import constants
from libs import helpers
from libs.metrics import helpers as helpers_metrics

def get_gt(aps_oa_data_tar_gz: str):
    # demographics
    df_all_authors_demographics = io.read_file_from_tar_gz_as_dataframe(aps_oa_data_tar_gz, constants.APS_OA_AUTHORS_DEMOGRAPHICS_FN)
    df_all_authors_demographics.rename(columns={'id_author':'id_author_oa'}, inplace=True)

    # scholarly stats
    df_all_authors_stats = io.read_file_from_tar_gz_as_dataframe(aps_oa_data_tar_gz, constants.APS_OA_AUTHORS_STATS_FN)
    df_all_authors_stats.rename(columns={'id_author':'id_author_oa'}, inplace=True)

    # gt (from APS)
    df_gt = df_all_authors_demographics[['id_author_oa','first_name','last_name','ethnicity','gender']].copy()
    df_gt = df_gt.merge(df_all_authors_stats[['id_author_oa','works_count','cited_by_count', 'rr1_rank_publications','rr1_rank_publications_percentile', 'rr2_rank_citations','rr2_rank_citations_percentile']], on='id_author_oa', how='left')
    df_gt = helpers.add_quantiles(df_gt)

    del df_all_authors_demographics
    del df_all_authors_stats
    return df_gt

def run(results_dir: str, model: str, metric: str, aps_oa_data_tar_gz: str, is_temperature_analysis: bool, output_dir: str, dont_save: bool, overwrite: bool):

    prefix = 'temperature' if is_temperature_analysis else None
    output_dir = io.path_join(output_dir, 'benchmarks')
    io.validate_path(output_dir)
    
    fn_results = helpers_metrics.get_per_attempt_fn(model, metric, output_dir, prefix=prefix)
    if io.exists(fn_results) and not overwrite:
        return io.read_csv(fn_results, index_col=0), None
    
    # Initialize tqdm progress bar
    is_factuality_metric = metric in constants.BENCHMARK_FACTUALITY_METRICS and metric != 'factuality_author'
    total_steps = 9 + (1 if is_factuality_metric else 0)

    progress_bar = tqdm(total=total_steps, desc="Loading data")

    # 1. process pre-processed data
    df_summaries_all = io.read_csv(io.path_join(results_dir, 'summaries', f"experiments_{model}.csv"), low_memory=False)
    progress_bar.update(1)
    
    df_valid_responses_all = io.read_csv(io.path_join(results_dir, 'valid_responses', f"{model}.csv"), index_col=0, low_memory=False)
    progress_bar.update(1)
    
    df_factuality_author_all = io.read_csv(io.path_join(results_dir, 'factuality', f"{model}_author.csv"), index_col=0, low_memory=False)
    progress_bar.update(1)

    df_factuality_task_all = None
    if is_factuality_metric:
        experiment_task = metric.split('_')[-1]
        df_factuality_task_all = io.read_csv(io.path_join(results_dir, 'factuality', f"{model}_{experiment_task}.csv"), index_col=0, low_memory=False)
        progress_bar.update(1)
        
    files = io.get_files(io.path_join(results_dir, 'similarities'), f"{model}_*.csv")
    df_similarity_all = io.pd.DataFrame()
    for fn in files:
        tmp = io.read_csv(fn, low_memory=False, index_col=0)
        df_similarity_all = io.pd.concat([df_similarity_all, tmp], ignore_index=True)
    progress_bar.update(1)


    # 2. filter out responses outside the intervention period
    if is_temperature_analysis:
        df_summaries = df_summaries_all.copy()
        df_valid_responses = df_valid_responses_all.copy()
        df_factuality_author = df_factuality_author_all.copy()
        df_factuality_task = df_factuality_task_all.copy() if df_factuality_task_all is not None else None
        df_similarity = df_similarity_all.copy()
    else:
        df_summaries = df_summaries_all.query(constants.INTERVENTION_PERIOD_QUERY).copy()
        df_valid_responses = df_valid_responses_all.query(constants.INTERVENTION_PERIOD_QUERY).copy()
        df_factuality_author = df_factuality_author_all.query(constants.INTERVENTION_PERIOD_QUERY).copy()
        df_factuality_task = df_factuality_task_all.query(constants.INTERVENTION_PERIOD_QUERY).copy() if df_factuality_task_all is not None else None
        df_similarity = df_similarity_all.query(constants.INTERVENTION_PERIOD_QUERY).copy()
    progress_bar.update(1)

    # 3. augmenting data

    # get gt 
    df_gt = get_gt(aps_oa_data_tar_gz)
    progress_bar.update(1)

    # adding prominence metrics to recommended authors
    df_factuality_author = df_factuality_author.merge(df_gt[['id_author_oa', 'prominence_pub', 'prominence_cit']], on='id_author_oa', how='left')
    progress_bar.update(1)

    # adding infrastructure metadata
    df_summaries = helpers.add_infrastructure_columns(df_summaries)
    df_factuality_author = helpers.add_infrastructure_columns(df_factuality_author)
    df_factuality_task = helpers.add_infrastructure_columns(df_factuality_task) if df_factuality_task is not None else None
    df_similarity = helpers.add_infrastructure_columns(df_similarity)
    df_valid_responses = helpers.add_infrastructure_columns(df_valid_responses)
    progress_bar.update(1)

    # 4. summarize metrics per attempt
    # computes the metric per attempt for ALL requests

    df = df_summaries if metric in constants.BENCHMARK_BINARY_METRICS else df_factuality_author
    gt = df_gt if metric in constants.BENCHMARK_PARITY_METRICS else None
    df_sim = df_similarity if metric in constants.BENCHMARK_SIMILARITY_METRICS else None
    metric_similarity = constants.BENCHMARK_SIMILARITY_METRICS_MAP.get(metric, None)

    print(f"Loaded {df_factuality_task.shape[0]} rows for {experiment_task} factuality")

    # aggregate metrics per attempt
    df_per_attempt = helpers_metrics.load_per_attempt(metric, df, fn_results,
                                                      save=not dont_save,
                                                      gt=gt, 
                                                      df_similarity=df_sim, 
                                                      df_factuality_task=df_factuality_task,
                                                      metric_similarity=metric_similarity,
                                                      overwrite=overwrite,
                                                      verbose=False)
    progress_bar.update(1)
    return df_per_attempt, fn_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str, help="Directory where the pre-processed data is stored (eg. results/interventions)")
    parser.add_argument("--model", required=True, type=str, choices=constants.LLMS, help=f"Model to analyse (ie., {', '.join(constants.LLMS)})")
    parser.add_argument("--metric", required=True, type=str, choices=constants.BENCHMARK_METRICS, help=f"Metric to analyse (ie., {', '.join(constants.BENCHMARK_METRICS)})")
    parser.add_argument("--aps_oa_data_tar_gz", required=True, type=str, help="final_dataset.tar.gz")
    parser.add_argument('--temperature_analysis', action='store_true', default=False, help="Whether the data is from the temperature analysis")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved (e.g., results/benchmarks)")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Whether to overwrite the existing output files")
    parser.add_argument("--dont_save", action='store_true', default=False, help="Whether to not save the output files")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    io.validate_path(args.output_dir)
    df_per_attempt, fn = run(args.results_dir, args.model, args.metric, args.aps_oa_data_tar_gz, args.temperature_analysis, args.output_dir, args.dont_save, args.overwrite)

    print(f"Data successfully saved to {fn}" if fn is not None else "Data already exists.") 
    print(f"Metric {args.metric} successfully processed for model {args.model}: {df_per_attempt.shape[0]} attempts")
    print(df_per_attempt.head(5))
    

    