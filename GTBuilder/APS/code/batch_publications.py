import argparse
from tqdm import tqdm

from libs import io
from libs import constants

def run(aps_os_data_tar_gz: str, output_dir: str):
    
    # publications
    df_pub = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_FN)
    df_pub['ID'] = range(1, len(df_pub) + 1)
    df_pub.rename(columns={'name':'display_name'}, inplace=True)
    
    # publications mapping with aps
    df_map = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_MAPPING_FN)

    # join aps id
    df_pub = df_pub.set_index('id_publication').join(df_map.set_index('oa_id')).reset_index()

    # converstion to list of dicts
    data = [get_object(row) for index, row in tqdm(df_pub.iterrows(), total=len(df_pub), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.PUBLICATIONS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)

def get_object(row):
    obj = {}
    obj['new_id_publication'] = row.ID
    obj['oa_id_publication'] = row.id_publication
    obj['aps_id_publication'] = row.aps_id
    obj['aps_id_journal'] = row.id_journal
    obj['doi'] = row.doi
    obj['title'] = row.title
    obj['oa_timestamp'] = row.publication_date
    obj['aps_timestamp'] = row.publication_date # @TODO: check if this is correct
    obj['language'] = row.language
    obj['oa_cited_by_count'] = row.cited_by_count
    
    return obj

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