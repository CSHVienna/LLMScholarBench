import argparse
from tqdm import tqdm

from libs import io
from libs import constants
from libs import helpers

def run(aps_os_data_tar_gz: str, output_dir: str):
    # authors
    df_author = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN)
    df_author.rename(columns={'id_author': 'id_author_oa'}, inplace=True)
    df_author['id_author'] = range(1, len(df_author) + 1)
    print(f'\n df_author: {df_author.shape} \n {df_author.head(2)} \n')

    # affiliations
    df_institution = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_INSTITUTIONS_FN)
    df_institution.rename(columns={'id_institution': 'id_institution_oa'}, inplace=True)
    df_institution['id_institution'] = range(1, len(df_institution) + 1)
    print(f'\n df_institution: {df_institution.shape} \n {df_institution.head(2)} \n')

    # author_institution_year
    df_author_inst_year = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_INSTITUTION_YEAR_FN)
    df_author_inst_year.rename(columns={'id_author': 'id_author_oa'}, inplace=True)
    df_author_inst_year.rename(columns={'id_institution': 'id_institution_oa'}, inplace=True)
    print(f'\n df_author_inst_year: {df_author_inst_year.shape} \n {df_author_inst_year.head(2)} \n')

    df_author_inst_year = df_author_inst_year.merge(df_author[['id_author_oa','id_author']], on='id_author_oa', how='left')
    print(f'\n df_author_inst_year (merge author): {df_author_inst_year.shape} \n {df_author_inst_year.head(2)} \n')
    
    df_author_inst_year = df_author_inst_year.merge(df_institution[['id_institution_oa','id_institution']], on='id_institution_oa', how='left')
    print(f'\n df_author_inst_year (merge institution): {df_author_inst_year.shape} \n {df_author_inst_year.head(2)} \n')

    data = [get_object(id_author_oa, id_author, group) for (id_author_oa,id_author), group in tqdm(df_author_inst_year.groupby(['id_author_oa','id_author']), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.AUTHORS_AFFILIATIONS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)


def get_object(id_author_oa, id_author, group):
    # {"id_affiliation": "af0000001", "openalex_institution_id": "https://openalex.org/I76130692", "years": [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]}
    affiliations = [{'id_affiliation':'' if helpers.is_none(id_institution) else int(id_institution), 
                     'openalex_institution_id':f'I{id_institution_oa}', 
                     'years':df.year.astype(int).sort_values().to_list()} for (id_institution_oa, id_institution), df in group.groupby(['id_institution_oa', 'id_institution'])]
    obj = {}
    obj['id_author'] = int(id_author)
    obj['openalex_author_id'] = f"A{id_author_oa}"
    obj['affiliations'] = affiliations
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