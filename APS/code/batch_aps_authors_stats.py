import argparse
from tqdm import tqdm
import pandas as pd

from libs import io
from libs import constants
from libs import scholar

tqdm.pandas()

def run(aps_os_data_tar_gz: str, aps_data_path: str, ranking: bool, output_dir: str):
    
    # 1. Read aps-oa data
    df_authors = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN) # id_author (oopenalex)
    df_authors['ID'] = range(1, len(df_authors) + 1)
    df_authors.rename(columns={'name':'display_name', 'id_author':'id_author_oa'}, inplace=True)
    print(f'\n df_authors: {df_authors.shape}  \n {df_authors.head(5)} \n')

    # 1.1 Read author mapping (mappring oa & aps)
    df_author_map = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_MAPPING_FN) # id_author_aps, id_author_oa 
    print(f'\n df_author_map: {df_author_map.shape}  \n {df_author_map.head(5)} \n')

    df_aps_author_names = io.read_csv(io.path_join(aps_data_path, constants.APS_AUTHOR_NAMES_FN)) # id_author, id_author_name
    df_aps_author_names.rename(columns={'id_author':'id_author_aps'}, inplace=True)
    df_aps_author_names = df_aps_author_names.merge(df_author_map[['id_author_oa', 'id_author_aps']], on='id_author_aps', how='left')
    print(f'\n df_aps_author_names: {df_aps_author_names.shape}  \n {df_aps_author_names.head(5)} \n')

    df_aps_publications = io.read_csv(io.path_join(aps_data_path, constants.APS_PUBLICATIONS_FN)) # id_publication, timestamp
    df_aps_publications['year'] = df_aps_publications.timestamp.str[:4].astype(int)
    print(f'\n df_aps_publications: {df_aps_publications.shape}  \n {df_aps_publications.head(5)} \n')

    df_aps_authorships = io.read_csv(io.path_join(aps_data_path, constants.APS_AUTHORSHIPS_FN)) # id_author_name, id_publication
    df_aps_authorships = df_aps_authorships.merge(df_aps_author_names[['id_author_oa','id_author_aps','id_author_name']], on='id_author_name', how='left')
    df_aps_authorships = df_aps_authorships.merge(df_aps_publications[['id_publication','year']], on='id_publication', how='left')
    print(f'\n df_aps_authorships: {df_aps_authorships.shape}  \n {df_aps_authorships.head(5)} \n')

    df_aps_citations = io.read_csv(io.path_join(aps_data_path, constants.APS_CITATIONS_FN)) # id_publication_citing A-->B id_publication_cited (A cites B; B receives a citation from A)
    df_aps_citations = df_aps_citations.merge(df_aps_authorships[['id_publication','id_author_aps','id_author_oa']], left_on='id_publication_cited', right_on='id_publication', how='left')
    print(f'\n df_aps_citations: {df_aps_citations.shape}  \n {df_aps_citations.head(5)} \n')

    # 3. Stats

    # work counts
    df_work_counts = df_aps_authorships.groupby('id_author_oa').id_publication.nunique().reset_index().rename(columns={'id_publication':'aps_works_count'})
    print(f'\n df_work_counts: {df_work_counts.shape}  \n {df_work_counts.head(5)} \n')

    # citations
    df_citation_counts = df_aps_citations.groupby('id_author_oa').id_publication_citing.count().reset_index().rename(columns={'id_publication_citing':'aps_cited_by_count'})
    print(f'\n df_citation_counts: {df_citation_counts.shape}  \n {df_citation_counts.head(5)} \n')

    # compute h-index
    col_name = 'citations'
    df_h_index = df_aps_citations.groupby('id_author_oa').apply(lambda group: scholar.compute_h_index(group.groupby('id_publication_cited').id_publication_citing.count().values)).reset_index(name='aps_h_index')
    print(f'\n df_h_index: {df_h_index.shape}  \n {df_h_index.head(5)} \n')

    # compute i10-index
    df_i10_index = df_aps_citations.groupby('id_author_oa').apply(lambda group: scholar.compute_i10_index(group.groupby('id_publication_cited').id_publication_citing.count().values)).reset_index(name='aps_i10_index')
    print(f'\n df_i10_index: {df_i10_index.shape}  \n {df_i10_index.head(5)} \n')

    # compute e-index
    df_e_index = df_aps_citations.groupby('id_author_oa').apply(lambda group: scholar.compute_e_index(group, col_publication='id_publication_cited', col_citation_from='id_publication_citing')).reset_index(name='aps_e_index')
    print(f'\n df_e_index: {df_e_index.shape}  \n {df_e_index.head(5)} \n')

    # time-related stats
    df_time = df_aps_authorships.groupby('id_author_oa').year.agg(['min','max']).reset_index().rename(columns={'min':'min_year','max':'max_year'})
    df_time['aps_years_of_activity'] = df_time.apply(lambda row: [row.min_year, row.max_year], axis=1)
    df_time.loc[:,'aps_career_age'] = df_time.apply(lambda row: row.max_year - row.df_time.min_year, axis=1) # TODO: add plus 1
    print(f'\n df_time: {df_time.shape}  \n {df_time.head(5)} \n')

    # 4. Final merge
    df_authors = df_authors.merge(df_work_counts[['id_author_oa','aps_works_count']], on='id_author_oa', how='left')
    df_authors = df_authors.merge(df_citation_counts[['id_author_oa','aps_cited_by_count']], on='id_author_oa', how='left')
    df_authors = df_authors.merge(df_time[['id_author_oa','aps_years_of_activity', 'aps_career_age']], on='id_author_oa', how='left')
    df_authors = df_authors.merge(df_h_index[['id_author_oa','aps_h_index']], on='id_author_oa', how='left')
    df_authors = df_authors.merge(df_i10_index[['id_author_oa','aps_i10_index']], on='id_author_oa', how='left')
    df_authors = df_authors.merge(df_e_index[['id_author_oa','aps_e_index']], on='id_author_oa', how='left')

    # 5. Null values
    df_authors.fillna({'aps_works_count':0, 'aps_cited_by_count':0, 'aps_h_index':0, 'aps_i10_index':0, 'aps_e_index':0, "aps_career_age": 0}, inplace=True) # TODO: this is momentary, we need to check why there are null values
    indexes_noyear = df_authors[df_authors['aps_years_of_activity'].isnull()].index
    df_authors.loc[indexes_noyear, 'aps_years_of_activity'] = df_authors.loc[indexes_noyear, 'aps_years_of_activity'].apply(lambda v: [])

    if not ranking:
        # Process json data
        data = [get_object(row) for index, row in tqdm(df_authors.iterrows(), total=len(df_authors), desc="Processing rows")]

        # Save json data
        fn = io.path_join(output_dir, constants.AUTHORS_APS_STATS_FN)
        io.save_dicts_to_text_file(dict_list=data, file_path=fn)
    else:
        ## RANKINGS:
        df_authors.loc[:,'aps_citations_per_paper_age'] = df_authors.progress_apply(lambda row: 0 if row.aps_works_count == 0 else (row.aps_cited_by_count/row.aps_works_count)/(row.aps_career_age+1), axis=1)

        # compute rankings
        for mid, (metric, col_name) in enumerate(constants.APS_RANKING_METRICS.items()):
            
            # Sort by Citations in descending order
            df_authors = df_authors.sort_values(by=col_name, ascending=False).reset_index(drop=True)

            # Rank scientists (using 'min' method to handle ties)
            col_rank = f"rr{mid+1}_rank_{metric}"
            df_authors[col_rank] = df_authors[col_name].rank(method='min', ascending=False)

            # Calculate Percentile
            col_perc = f"rr{mid+1}_rank_{metric}_percentile"
            total_scientists = len(df_authors)
            df_authors[col_perc] = (1 - (df_authors[col_rank] - 1) / total_scientists) * 100
            print(f'\n df_author ({col_perc}): {df_authors.shape}  \n {df_authors.head(2)} \n')

        # Process json data
        df_authors.sort_values('ID', ascending=True, inplace=True)
        data = [get_ranking_object(row) for index, row in tqdm(df_authors.iterrows(), total=len(df_authors), desc="Processing rows")]

        # Save json data
        fn = io.path_join(output_dir, constants.AUTHORS_APS_RANKINGS_FN)
        io.save_dicts_to_text_file(dict_list=data, file_path=fn)


def get_object(row):
    obj = {}
    obj['id_aps_stat'] = row.ID
    obj['id_author'] = row.ID
    obj['openalex_id'] = f"A{row.id_author_oa}" #not needed but as reference
    obj['aps_works_count'] = row.aps_works_count
    obj['aps_cited_by_count'] = row.aps_cited_by_count
    obj['aps_h_index'] = row.aps_h_index
    obj['aps_i10_index'] = row.aps_i10_index
    obj['aps_e_index'] = row.aps_e_index
    obj['aps_years_of_activity'] = row.aps_years_of_activity
    obj['aps_career_age'] = row.aps_career_age

    return obj

def get_ranking_object(row):

    obj = {}
    obj['id_author_ranking'] = row.ID
    obj['id_author'] = row.ID
    obj['openalex_id'] = f"A{row.id_author_oa}"
    
    for mid, (metric, col_name) in enumerate(constants.APS_RANKING_METRICS.items()):
        
        col_rank = f"rr{mid+1}_rank_{metric}"
        col_perc = f"rr{mid+1}_rank_{metric}_percentile"

        obj[col_rank] = row[col_rank]
        obj[col_perc] = row[col_perc]

    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", type=str, help="final_dataset.tar.gz")
    parser.add_argument("--aps_data_path", type=str, help="Directory where the original aps data files are")
    parser.add_argument("--ranking", action="store_true", help="Flag to indicate if the rankings should be computed")
    parser.add_argument("--output_dir", type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.aps_data_path, args.ranking, args.output_dir)