import argparse
from tqdm import tqdm

from libs import io
from libs import constants
from libs import helpers

def run(aps_os_data_tar_gz: str, output_dir: str):
    # affiliations
    df_institution = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_INSTITUTIONS_FN)
    df_institution['ID'] = range(1, len(df_institution) + 1)
    df_institution.rename(columns={'id_institution': 'id_institution_oa'}, inplace=True)
    print(f'\n df_institution: {df_institution.shape} \n {df_institution.head(2)} \n')

    data = [get_object(row) for index, row in tqdm(df_institution.iterrows(), total=len(df_institution), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.INSTITUTIONS_STATS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)


def get_object(row):
    # id_institution,cited_by_count,country_code,created_date,updated_date,display_name,display_name_acronyms,ror,2yr_mean_citedness,h_index,i10_index,type,works_count,city
    # {"id_institution_stat": "is000001", "id_affiliation": "af0044400", "h_index": 22, "mean_citedness_2yr": 2.3333333333, "i10_index": 40, "cited_by_count": 1664, "works_count": 78}
    obj = {}
    obj['id_institution_stat'] = row.ID
    obj['id_affiliation'] = row.ID
    obj['id_affiliation_oa'] = f"I{row.id_institution_oa}" # not needed but just in case
    obj['h_index'] = row.h_index
    obj['mean_citedness_2yr'] = row['2yr_mean_citedness']
    obj['i10_index'] = row.i10_index
    obj['cited_by_count'] = row.cited_by_count
    obj['works_count'] = row.works_count

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

    