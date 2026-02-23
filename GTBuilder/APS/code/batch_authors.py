import argparse
from tqdm import tqdm

from libs import io
from libs import constants

def run(aps_os_data_tar_gz: str, output_dir: str):
    
    # Do something with the input files
    df = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN)
    df['ID'] = range(1, len(df) + 1)
    df.rename(columns={'name':'display_name'}, inplace=True)
    data = [get_object(row) for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.AUTHORS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)

def get_object(row):
    obj = {}
    obj['id_author'] = row.ID
    obj['openalex_id'] = f"A{row.id_author}"
    obj['display_name'] = row.display_name
    obj['created_date'] = row.created_date
    obj['updated_date'] = row.updated_date
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