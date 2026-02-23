import argparse
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from functools import partial

from libs import io
from libs import constants
from libs import ethnicity
from libs import gender
from libs import parallel

def get_longest_name(lst): # TODO: Move to helpers.py
    cleaned_list = [item for item in lst if item is not None and item != ''  and not pd.isna(item)]
    if isinstance(cleaned_list, list) and len(cleaned_list) > 0 and cleaned_list is not None:
        return max(cleaned_list, key=len)
    return None

N_CHUNKS = 10

def run(aps_os_data_tar_gz: str, aps_gender_fn:str, intermediate_output_dir:str, task: str):
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
    df_authors[col_longestname] = df_authors['alternative_names'].apply(get_longest_name) # TODO: Use from helpers.py
    print(f'\n df_authors (longest name): {df_authors.shape}  \n {df_authors.head(5)} \n')    

    # Step 8. Splitting first and last name
    col_firstname = 'first_name'
    col_lastname = 'last_name'
    for (col_name, index) in [(col_lastname, -1), (col_firstname,0)]:
        if col_name not in df_authors:
            df_authors[col_name] = df_authors[col_longestname].apply(lambda x: x.split()[index])
    print(f'\n df_authors (first, last name): {df_authors.shape}  \n {df_authors.head(5)} \n')    

    print(f"\n {col_lastname}: {df_authors[col_lastname].nunique()} ({df_authors.shape[0]} records)\n")
    
    if task == new_col_dx:
        # Ethinicity: demographicx (full name)    
        if io.exists(fn_dx):
            print(f"Reading: {fn_dx}")
            df_authors = df_authors.merge(io.read_csv(fn_dx, index_col=None), on=author_index_col, how='left')
        else:
            df_authors = ethnicity.dx_predict_ethnicity(df_authors, col_longestname, new_col_dx)
            df_authors[new_col_dx] = df_authors[new_col_dx].replace(constants.DEMOGRAPHICX_ETHNICITY_MAPPING).fillna(constants.UNKNOWN_STR)
            io.save_csv(df_authors[[author_index_col] + [new_col_dx]], fn_dx, index=False)
        print(f'\n df_authors (ethnicity_dx): {df_authors.shape}  \n {df_authors.head(5)} \n')

    elif task == new_col_ec:    
        # Ethinicity: Ethinicolor (last name)
        if io.exists(fn_ec):
            print(f"Reading: {fn_ec}")
            df_authors = df_authors.merge(io.read_csv(fn_ec, index_col=None), on=author_index_col, how='left')
        else:
            df_authors = ethnicity.ec_predict_ethnicity(df_authors, col_lastname, new_col_ec)
            df_authors[new_col_ec] = df_authors[new_col_ec].replace(constants.DEMOGRAPHICX_ETHNICITY_MAPPING).fillna(constants.UNKNOWN_STR)
            io.save_csv(df_authors[[author_index_col] + [new_col_ec]], fn_ec, index=False)
        print(f'\n df_authors (ethnicity_ec): {df_authors.shape}  \n {df_authors.head(5)} \n')

    elif task == col_gender:
        
        # Fallback approach
        if io.exists(fn_ethnicity):
            print(f"Reading: {fn_ethnicity}")
            df_authors = df_authors.merge(io.read_csv(fn_ethnicity, index_col=None), on=author_index_col, how='left')
        else:
            if io.exists(fn_dx) and io.exists(fn_ec):
                df_authors = df_authors.merge(io.read_csv(fn_dx, index_col=None), on=author_index_col, how='left')
                df_authors = df_authors.merge(io.read_csv(fn_ec, index_col=None), on=author_index_col, how='left')
                df_authors[col_ethnicity] = df_authors.apply(lambda row: ethnicity.choose_ethnicity(row, new_col_dx, new_col_ec), axis=1)
                io.save_csv(df_authors[[author_index_col] + [col_ethnicity]], fn_ethnicity, index=False)
                print(f'\n df_authors (ethnicity): {df_authors.shape}  \n {df_authors.head(5)} \n')
            else:
                raise FileNotFoundError(f"Files '{new_col_dx}' and '{new_col_ec}' not found.")

        ### GENDER        
        if io.exists(fn_gender):
            print(f"Reading: {fn_gender}")
            df_authors = df_authors.merge(io.read_csv(fn_gender, index_col=None), on=author_index_col, how='left')
        else:
            
            processed_results = parallel_processing(df_authors, fn_gender, col_gender, col_ethnicity, col_firstname, author_index_col, n_chunks=N_CHUNKS)
            df_gender = pd.concat(processed_results, ignore_index=True)
            df_authors = df_authors.merge(df_gender, on=author_index_col, how='left')

        #df_authors[col_gender] = df_authors.progress_apply(lambda row: gender.assign_combined_gender(row, col_ethnicity, col_gender, col_firstname), axis=1).fillna(constants.UNKNOWN_STR)
        io.save_csv(df_authors[[author_index_col] + [col_gender]], fn_gender, index=False)

        print(f'\n df_authors (gender): {df_authors.shape}  \n {df_authors.head(5)} \n')
    
    else:
        raise ValueError(f"Invalid task: {task}")


def process_chunk(chunk_data, fn_gender, col_gender, col_ethnicity, col_firstname, author_index_col):

    chunk, chunk_index = chunk_data

    fn_gender_chunk = fn_gender.replace('.csv',f'_{chunk_index}.csv')
    if io.exists(fn_gender_chunk):
        print(f"Reading: {fn_gender_chunk}")
        chunk = io.read_csv(fn_gender_chunk)
    else:
        chunk[col_gender] = chunk.progress_apply(lambda row: gender.assign_combined_gender(row, col_ethnicity, col_gender, col_firstname), axis=1).fillna(constants.UNKNOWN_STR)
        io.save_csv(chunk[[author_index_col] + [col_gender]], fn_gender_chunk, index=False)
        print(f"Saved: {fn_gender_chunk}")

    return chunk

def parallel_processing(df, fn_gender, col_gender, col_ethnicity, col_firstname, author_index_col, n_chunks=10):
    # Split the DataFrame into n_chunks
    chunks = [(df.iloc[i::n_chunks].copy(), i) for i in range(n_chunks)]

    process_chunk_with_params = partial(process_chunk, fn_gender=fn_gender, col_gender=col_gender, col_ethnicity=col_ethnicity, col_firstname=col_firstname, author_index_col=author_index_col)

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=n_chunks) as pool:
        results = list(tqdm(pool.imap(process_chunk_with_params, chunks), total=n_chunks))
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", type=str, help="final_dataset.tar.gz")
    parser.add_argument("--aps_gender_fn", type=str, help="author_names.csv")
    parser.add_argument("--intermediate_output_dir", type=str, help="Directory where the intermediate files will be saved")
    parser.add_argument("--task", type=str, choices=['ethnicity_dx','ethnicity_ec','gender'], help="What to do ethnicity_dx, ethnicity_ec, or gender")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.aps_gender_fn, args.intermediate_output_dir, args.task)