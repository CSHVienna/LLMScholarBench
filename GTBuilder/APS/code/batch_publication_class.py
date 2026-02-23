import argparse
from tqdm import tqdm
import pandas as pd

from libs import io
from libs import constants
from libs import helpers

def run(aps_os_data_tar_gz: str, aps_data_path: str, output_dir: str):
    
    # 1. Read aps-oa data
    df_pub = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_FN) # id_author (oopenalex)
    df_pub['ID'] = range(1, len(df_pub) + 1)
    df_pub.rename(columns={'id_publication':'id_publication_oa'}, inplace=True)
    print(f'\n df_pub: {df_pub.shape}  \n {df_pub.head(5)} \n')

    # publications mapping with aps
    df_map = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_MAPPING_FN)
    df_map.rename(columns={'oa_id':'id_publication_oa', 'aps_id':'id_publication_aps'}, inplace=True)
    print(f'\n df_map: {df_map.shape}  \n {df_map.head(5)} \n')

    # join aps id
    df_pub = df_pub.merge(df_map[['id_publication_oa','id_publication_aps']], on='id_publication_oa').reset_index()
    print(f'\n df_pub (merge map): {df_pub.shape}  \n {df_pub.head(5)} \n')

    # 2. Read APS data

    # topics: concept, area, discipline
    df_topic_types = io.read_csv(io.path_join(aps_data_path, constants.APS_TOPIC_TYPES_FN)) # id_topic_type, name
    df_topic_types.rename(columns={'name':'type_name'}, inplace=True)
    print(f'\n df_topic_types: {df_topic_types.shape}  \n {df_topic_types.head(5)} \n')

    # publication-topic
    df_publication_topics = io.read_csv(io.path_join(aps_data_path, constants.APS_PUBLICTION_TOPICS)) # id_publication, id_topic, id_topic_type, primary
    df_publication_topics.rename(columns={'id_publication':'id_publication_aps'}, inplace=True)
    df_publication_topics = df_publication_topics.merge(df_topic_types[['id_topic_type','type_name']], on='id_topic_type', how='left')
    df_publication_topics.rename(columns={'id_publication':'id_publication_aps'}, inplace=True)
    print(f'\n df_publication_topics: {df_publication_topics.shape}  \n {df_publication_topics.head(5)} \n')

    # 4. Mergin APS data to OA
    df_pub = df_pub.merge(df_publication_topics[['id_publication_aps','id_topic','id_topic_type','type_name','primary']], on='id_publication_aps', how='left').drop(columns=['index'])
    df_pub['id_publication_classification'] = range(1, len(df_pub) + 1)
    print(f'\n df_pub (final merge): {df_pub.shape}  \n {df_pub.head(5)} \n')

    # Process json data
    data = [get_object(row) for index, row in tqdm(df_pub.iterrows(), total=len(df_pub), desc="Processing rows")]

    # Save json data
    fn = io.path_join(output_dir, constants.PUBLICATION_CLASS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)

def get_object(row):
    obj = {}
    obj['id_publication_classification'] = row.id_publication_classification
    obj['new_id_publication'] = row.ID
    obj['aps_id_publication'] = row.id_publication_aps
    obj['oa_id_publication'] = row.id_publication_oa # not needed but just in case
    obj['aps_id_topic'] = '' if helpers.is_none(row.id_topic) else int(row.id_topic)
    obj['aps_id_topic_type'] = '' if helpers.is_none(row.id_topic_type) else int(row.id_topic_type)
    obj['id_topic'] = '' if helpers.is_none(row.type_name) or helpers.is_none(row.id_topic) else f"{row.type_name[0]}_{int(row.id_topic):07}"
    obj['classification_type'] = '' if helpers.is_none(row.type_name) else row.type_name
    obj['primary'] = False if helpers.is_none(row.type_name) else row.primary in [constants.TRUE_STR, True]
    
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", type=str, help="final_dataset.tar.gz")
    parser.add_argument("--aps_data_path", type=str, help="Directory where the original aps data files are")
    parser.add_argument("--output_dir", type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.aps_data_path, args.output_dir)