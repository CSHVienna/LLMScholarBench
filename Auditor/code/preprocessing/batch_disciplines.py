# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from operator import index
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from functools import partial

from libs import experiments
from libs import io
from libs import constants
from libs import text

def run(aps_os_data_tar_gz: str, aps_data_dir: str, output_dir: str):
    # APS data
    df_publication_topic = io.read_csv(io.path_join(aps_data_dir, constants.APS_PUBLICTION_TOPICS))
    df_disciplines = io.read_csv(io.path_join(aps_data_dir, constants.APS_DISCIPLINES_FN))
    #df_topic_types = io.read_csv(io.path_join(aps_data_dir, constants.APS_TOPIC_TYPES_FN))
    df_authorships = io.read_csv(io.path_join(aps_data_dir, constants.APS_AUTHORSHIPS_FN))
    df_author_names = io.read_csv(io.path_join(aps_data_dir, constants.APS_AUTHOR_NAMES_FN))
    print("1a. df_authorships: ", df_authorships.shape, df_authorships.id_author_name.nunique())

    # APS OA data
    df_author_mapping = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_MAPPING_FN)
    df_author_demographics = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_DEMOGRAPHICS_FN)
    df_author_demographics.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    print("1b. df_author_mapping: ", df_author_mapping.shape, df_author_mapping.id_author_aps.nunique())

    # mergins
    df_authorships = df_authorships.merge(df_author_names, on='id_author_name', how='left')
    print("2. df_authorships: ", df_authorships.shape, df_authorships.id_author_name.nunique())

    df_authorships = df_authorships.merge(df_publication_topic, on='id_publication', how='left')
    print("3. df_authorships: ", df_authorships.shape, df_authorships.id_author_name.nunique())

    df_authorships = df_authorships.merge(df_disciplines, left_on='id_topic', right_on='id_discipline', how='left')
    print("4. df_authorships: ", df_authorships.shape, df_authorships.id_author_name.nunique())

    df_author_mapping = df_author_mapping.merge(df_authorships, right_on='id_author', left_on='id_author_aps', how='left')
    print("5. df_author_mapping: ", df_author_mapping.shape, df_author_mapping.id_author_aps.nunique())

    df_author_mapping = df_author_mapping.merge(df_author_demographics, on='id_author_oa', how='left')
    print("6. df_author_mapping: ", df_author_mapping.shape, df_author_mapping.id_author_aps.nunique())

    ### NOT IN DISCIPLINES
    df_data_not = clean_data(df_author_mapping, False)
    result_not_g, result_not_e = calculate_basic_stats(df_data_not)
    df_data_not.shape, result_not_g.shape, result_not_e.shape
    io.printf(f"Authors not in disciplines: {df_data_not.shape}")
    print(result_not_g)
    print(result_not_e)
    print()

    ### DISCIPLINES:
    df_data_in = clean_data(df_author_mapping, True)
    result_in = calculate_stats_by_discipline(df_data_in)
    io.printf(f"Authors in disciplines: {df_data_in.shape}")
    print(result_in)

    fn = io.path_join(output_dir, constants.METADATA_DIR, constants.APS_OA_DISCIPLINES_DEMOGRAPHICS_FN)
    io.validate_path(fn)
    io.save_csv(result_in, fn)


def clean_data(df, disciplines=True):
    ineq = '==' if disciplines else '!=' 
    df_data = df.query(f"id_topic_type {ineq} @constants.APS_TOPIC_DISCIPLINE_ID").copy()
    cols = ['id_discipline','label','id_author_oa'] if disciplines else ['id_author_oa']
    df_data.drop_duplicates(subset=cols, inplace=True)
    df_data = df_data[['id_author_oa', 'gender', 'ethnicity'] + (['label'] if disciplines else [])]
    return df_data

def calculate_basic_stats(df):
    # Calculate counts for gender
    gender_counts = df["gender"].value_counts().rename("gender_count")
    gender_fractions = (df["gender"].value_counts(normalize=True) * 100).rename("gender_fraction")
    stats_gender = io.pd.concat([gender_counts, gender_fractions], axis=1)

    # Calculate counts for ethnicity
    ethnicity_counts = df["ethnicity"].value_counts().rename("ethnicity_count")
    ethnicity_fractions = (df["ethnicity"].value_counts(normalize=True) * 100).rename("ethnicity_fraction")
    stats_ethnicity = io.pd.concat([ethnicity_counts, ethnicity_fractions], axis=1)
    return stats_gender, stats_ethnicity

def calculate_stats_by_discipline(df):
    # Calculate unique authors per discipline
    unique_authors = df.groupby(["label"])["id_author_oa"].nunique().rename("unique_authors")

    # Gender stats: unique counts and fractions
    gender_stats = (
        df.groupby(["label", "gender"])["id_author_oa"]
        .nunique()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
    )
    gender_fractions = gender_stats.div(gender_stats.sum(axis=1), axis=0)

    # Ethnicity stats: unique counts and fractions
    ethnicity_stats = (
        df.groupby(["label", "ethnicity"])["id_author_oa"]
        .nunique()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
    )
    ethnicity_fractions = ethnicity_stats.div(ethnicity_stats.sum(axis=1), axis=0)

    # Combine results into a single DataFrame
    result = io.pd.concat(
        {
            "unique_authors": unique_authors,
            "gender_counts": gender_stats,
            "gender_fractions": gender_fractions,
            "ethnicity_counts": ethnicity_stats,
            "ethnicity_fractions": ethnicity_fractions,
        },
        axis=1
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", required=True, type=str, help="final_dataset.tar.gz")
    parser.add_argument("--aps_data_dir", required=True, type=str, help="Directory to APS original data")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.aps_data_dir, args.output_dir)

    