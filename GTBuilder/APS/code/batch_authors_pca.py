import argparse
from tqdm import tqdm

from libs import io
from libs import constants
from libs.pca_similarity import PCASimilarityModel
from libs.pca_similarity import fit_pca_similarity

def run(aps_os_data_tar_gz: str, output_dir: str):
    
    # OpenAlex stats
    df_author_stats = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_STATS_FN)
    df_author_stats.rename(columns={'id_author': 'id_author_oa'}, inplace=True)
    print(f'\n df_author_stats: {df_author_stats.shape}  \n {df_author_stats.head(5)} \n')

    # APS stats
    df_author_aps_stats = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHOR_STATS_FN)
    df_author_aps_stats.rename(columns={'id_author': 'id_author_oa'}, inplace=True)
    df_author_aps_stats.shape

    # col metrics
    stats_columns = ['two_year_mean_citedness', 'h_index', 'i10_index', 'works_count',
                 'cited_by_count', 'e_index', 'career_age', 'max_year', 'min_year',
                 'citations_per_paper_age'] #10
    stats_aps_columns = ['aps_works_count', 'aps_cited_by_count', 'aps_h_index',
                     'aps_i10_index', 'aps_e_index', 'aps_years_of_activity',
                     'aps_career_age', 'aps_citations_per_paper_age'] #8

    # merge stats
    df_author_stats_all = df_author_stats.set_index('id_author_oa').join(df_author_aps_stats.set_index('id_author_oa'), how='left').copy()
    df_author_stats_all.reset_index(inplace=True)
    print(f'\n df_author_stats_all: {df_author_stats_all.shape}  \n {df_author_stats_all.head(5)} \n')

    # Run PCA
    cols = stats_columns + stats_aps_columns
    model = fit_pca_similarity(df_author_stats_all[['id_author_oa'] + cols], 
                            id_col="id_author_oa", 
                            explained_variance_target=0.9, 
                            use_log1p=True,
                            l2_normalize_embedding=True,
                            compute_rbf_sigma=False)

    # Save model
    model_fn = io.path_join(output_dir, constants.APS_OA_PCA_MODEL_FN)
    model.save_h5(model_fn)
    print(f'\n Model saved to: {model_fn} \n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", type=str, help="final_dataset.tar.gz")
    parser.add_argument("--output_dir", type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.output_dir)