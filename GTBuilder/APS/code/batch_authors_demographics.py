import argparse
from tqdm import tqdm
import pandas as pd

from libs import io
from libs import constants
from libs import ethnicity
from libs import gender
from libs import helpers


def run(aps_os_data_tar_gz: str, aps_gender_fn:str, intermediate_output_dir:str, output_dir: str):
    new_col_dx = 'ethnicity_dx'
    new_col_ec = 'ethnicity_ec'
    col_ethnicity = 'ethnicity'
    col_gender = 'gender'
    fn_dx = io.path_join(intermediate_output_dir, f"{new_col_dx}.csv")
    fn_ec = io.path_join(intermediate_output_dir, f"{new_col_ec}.csv")
    fn_ethnicity = io.path_join(intermediate_output_dir, f"{col_ethnicity}.csv")
    fn_gender = io.path_join(intermediate_output_dir, f"{col_gender}.csv")

    # 1. Authors
    author_index_col = 'id_author'
    df_authors = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN)
    df_authors['ID'] = range(1, len(df_authors) + 1)
    df_authors.rename(columns={'name':'display_name'}, inplace=True)
    print(f'\n df_authors: {df_authors.shape} \n {df_authors.head(5)} \n')



    # 2. Authors mapping with APS
    df_map = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_MAPPING_FN)
    print(f'\n df_map: {df_map.shape} \n {df_map.head(5)} \n')



    # 3. Gender (NQ)
    df_gender = io.read_csv(aps_gender_fn)
    col_gender_nq = 'gender_nq'
    df_gender.rename(columns={'id_author':'id_author_aps'}, inplace=True)
    df_gender = df_gender.groupby('id_author_aps')[col_gender_nq].apply(lambda group: constants.GENDER_UNISEX if 'gm' in group.values and 'gf' in group.values else 
                                                                                   constants.GENDER_MALE if 'gm' in group.values else 
                                                                                   constants.GENDER_FEMALE if 'gf' in group.values else 
                                                                                   constants.UNKNOWN_STR).reset_index()
    print(f'\n df_gender:  {df_gender.shape} \n {df_gender.head(5)} \n')



    # Step 4. Merging
    merge_oa_aps = pd.merge(df_authors, df_map[['id_author_aps','id_author_oa']], left_on='id_author', right_on='id_author_oa', how='left')
    print(f'\n merge_oa_aps:  {merge_oa_aps.shape} \n {merge_oa_aps.head(5)} \n')

    merged_df = pd.merge(merge_oa_aps, df_gender[['id_author_aps','gender_nq']], on='id_author_aps', how='left')
    print(f'\n merged_df:  {merged_df.shape} \n {merged_df.head(5)} \n')

    gender_summary = merged_df.groupby('id_author').apply(lambda group: gender.determine_gender(group, col_gender_nq)).reset_index(name=col_gender_nq)
    print(f'\n gender_summary:  {gender_summary.shape} \n {gender_summary.head(5)} \n')





    # Step 5: Merge the result back into df1
    df_authors = pd.merge(df_authors, gender_summary, on='id_author', how='left')
    print(f'\n df_authors (final):  {df_authors.shape} \n {df_authors.head(5)} \n')
    


    # Step 6. Alternative names 
    df_names = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_ALTERNATIVE_NAMES_FN)
    df_grouped_names = df_names.groupby('id_author')['alternative_name'].apply(list).reset_index()
    print(f'\n df_grouped_names: \n {df_grouped_names.head(2)} \n')
    
    df_authors = pd.merge(df_authors, df_grouped_names, on='id_author', how='left')
    print(f'\n df_authors (alternative names):  {df_authors.shape}  \n {df_authors.head(5)} \n')



    # Step 7. Longest name
    col_longestname = 'longest_name'
    df_authors.rename(columns={'alternative_name': 'alternative_names'}, inplace=True)
    df_authors[col_longestname] = df_authors['alternative_names'].apply(helpers.get_longest_name)
    print(f'\n df_authors (longest name): {df_authors.shape}  \n {df_authors.head(5)} \n')    



    # Step 8. Splitting first and last name
    col_firstname = 'first_name'
    col_lastname = 'last_name'
    for (col_name, index) in [(col_lastname, -1), (col_firstname,0)]:
        if col_name not in df_authors:
            df_authors[col_name] = df_authors[col_longestname].apply(lambda x: x.split()[index])
    print(f'\n df_authors (first, last name): {df_authors.shape}  \n {df_authors.head(5)} \n')    

    print(f"\n {col_lastname}: {df_authors[col_lastname].nunique()} ({df_authors.shape[0]} records)\n")


    ### ETHNICITY
    # Ethinicity: demographicx (full name)
    if io.exists(fn_dx):
        print(f"Reading: {fn_dx}")
        df_authors = df_authors.merge(io.read_csv(fn_dx, index_col=None), on=author_index_col, how='left')
    else:
        df_authors = ethnicity.dx_predict_ethnicity(df_authors, col_longestname, new_col_dx)
        df_authors[new_col_dx] = df_authors[new_col_dx].replace(constants.DEMOGRAPHICX_ETHNICITY_MAPPING).fillna(constants.UNKNOWN_STR)
        # io.save_csv(df_authors[[author_index_col] + [new_col_dx]], fn_dx, index=False)
    print(f'\n df_authors (ethnicity_dx): {df_authors.shape}  \n {df_authors.head(5)} \n')
    
    # Ethinicity: Ethinicolor (last name)
    if io.exists(fn_ec):
        print(f"Reading: {fn_ec}")
        df_authors = df_authors.merge(io.read_csv(fn_ec, index_col=None), on=author_index_col, how='left')
    else:
        df_authors = ethnicity.ec_predict_ethnicity(df_authors, col_lastname, new_col_ec)
        df_authors[new_col_ec] = df_authors[new_col_ec].replace(constants.DEMOGRAPHICX_ETHNICITY_MAPPING).fillna(constants.UNKNOWN_STR)
        # io.save_csv(df_authors[[author_index_col] + [new_col_ec]], fn_ec, index=False)
    print(f'\n df_authors (ethnicity_ec): {df_authors.shape}  \n {df_authors.head(5)} \n')

    # Fallback approach
    if io.exists(fn_ethnicity):
        print(f"Reading: {fn_ethnicity}")
        df_authors = df_authors.merge(io.read_csv(fn_ethnicity, index_col=None), on=author_index_col, how='left')
    else:
        df_authors[col_ethnicity] = df_authors.apply(lambda row: ethnicity.choose_ethnicity(row, new_col_dx, new_col_ec), axis=1)
        # io.save_csv(df_authors[[author_index_col] + [col_ethnicity]], fn_ethnicity, index=False)
    print(f'\n df_authors (ethnicity): {df_authors.shape}  \n {df_authors.head(5)} \n')





    ### GENDER
    if io.exists(fn_gender):
        print(f"Reading: {fn_gender}")
        df_authors = df_authors.merge(io.read_csv(fn_gender, index_col=None), on=author_index_col, how='left')
    else:
        df_authors[col_gender] = df_authors.progress_apply(lambda row: gender.assign_combined_gender(row, col_ethnicity, col_gender, col_firstname), axis=1).fillna(constants.UNKNOWN_STR)
        # io.save_csv(df_authors[[author_index_col] + [col_gender]], fn_gender, index=False)
    print(f'\n df_authors (gender): {df_authors.shape}  \n {df_authors.head(5)} \n')





    # Save df_authors 
    fn = io.path_join(intermediate_output_dir, constants.APS_OA_AUTHORS_DEMOGRAPHICS_FN) 
    io.save_csv(df_authors.drop(columns=['two_year_mean_citedness','h_index','i10_index','works_count','cited_by_count','ID']), fn, index=False) # remove the columns that are not needed (stats)




    # Process final data
    data = [get_object(row) for index, row in tqdm(df_authors.iterrows(), total=len(df_authors), desc="Processing rows")]

    # Save
    fn = io.path_join(output_dir, constants.AUTHORS_DEMOGRAPHICS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)


def get_object(row):
    obj = {}
    obj['id_author'] = row.ID
    obj['openalex_id'] = f"A{row.id_author}"
    obj['display_name'] = row.display_name
    obj['alternative_names'] = row.alternative_names
    obj['gender'] = row.gender
    obj['ethnicity'] = row.ethnicity
    obj['longest_name'] = row.longest_name

    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", type=str, help="final_dataset.tar.gz")
    parser.add_argument("--aps_gender_fn", type=str, help="author_names.csv")
    parser.add_argument("--intermediate_output_dir", type=str, help="Directory where the intermediate files will be saved")
    parser.add_argument("--output_dir", type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.aps_gender_fn, args.intermediate_output_dir, args.output_dir)