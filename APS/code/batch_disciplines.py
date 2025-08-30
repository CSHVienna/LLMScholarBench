import argparse
from tqdm import tqdm

from libs import io
from libs import constants

def run(aps_data_path: str, output_dir: str):
    
    # publications
    df_dis = io.read_csv(io.path_join(aps_data_path, constants.APS_DISCIPLINES_FN))
    df_dis['ID'] = range(1, len(df_dis) + 1)
    
    # converstion to list of dicts
    data = [get_object(row) for index, row in tqdm(df_dis.iterrows(), total=len(df_dis), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.DISCIPLINES_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)

def get_object(row):
    obj = {}
    obj['new_id_discipline'] = row.ID
    obj['aps_id_discipline'] = row.id_discipline
    obj['code'] = row.code
    obj['label'] = row.label
    
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_data_path", type=str, help="Directory where the original aps data files are")
    parser.add_argument("--output_dir", type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_data_path, args.output_dir)