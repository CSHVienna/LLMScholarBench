import argparse
from tqdm import tqdm

from libs import io
from libs import constants

def run(aps_os_data_tar_gz: str, output_dir: str):
    
    # auhtors
    df_author = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN)
    df_author.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    df_author['id_author'] = range(1, len(df_author) + 1)
    
    # collaborations
    df_authorship = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORSHIPS_FN)
    df_authorship.rename(columns={'id_author':'id_author_oa'}, inplace=True)
    df_authorship['id_network'] = range(1, len(df_authorship) + 1)
    df_authorship = df_authorship.merge(df_author[['id_author_oa','id_author']], on='id_author_oa', how='left')

    data = [get_object(row.id_author, row.id_author_oa, df_authorship) for index, row in tqdm(df_author.iterrows(), total=len(df_author), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.COAUTHOR_NETWORKS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)

def get_object(id_author, id_author_oa, df_authorship):

    id_publications = df_authorship.query("id_author_oa == @id_author_oa").id_publication.unique()
    coauthors = df_authorship.query("id_publication in @id_publications").id_author.dropna().astype(int).values.tolist()
    coauthors = list(set(coauthors) - set([id_author]))

    obj = {}
    obj['id_network'] = id_author
    obj['id_author'] = id_author
    obj['openalex_id'] = f"A{id_author_oa}"
    obj['aps_co_authors'] = coauthors
    obj['collaboration_counts'] = len(coauthors)

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