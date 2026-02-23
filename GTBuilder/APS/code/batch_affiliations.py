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
    df_institution['id_institution'] = range(1, len(df_institution) + 1)
    print(f'\n df_institution: {df_institution.shape} \n {df_institution.head(2)} \n')

    data = [get_object(row) for index, row in tqdm(df_institution.iterrows(), total=len(df_institution), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.AFFILIATIONS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)


def get_object(row):
    # id_institution,cited_by_count,country_code,created_date,updated_date,display_name,display_name_acronyms,ror,2yr_mean_citedness,h_index,i10_index,type,works_count
    # {"id_affiliation": "af0000001", "openalex_id": "https://openalex.org/I76130692", "display_name": "Zhejiang University", "type": "education", "country_code": "CN", "ror": "https://ror.org/00a2xv884", "city": "Hangzhou"}
    obj = {}
    obj['id_affiliation'] = row.ID
    obj['openalex_id'] = f"I{row.id_institution_oa}"
    obj['display_name'] = row.display_name
    obj['type'] = row.type
    obj['country_code'] = row.country_code
    obj['ror'] = row.ror
    obj['city'] = row.city

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

    