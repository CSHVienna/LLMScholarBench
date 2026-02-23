import argparse
from tqdm import tqdm

from libs import io
from libs import constants
from libs import scholar

def run(aps_os_data_tar_gz: str, output_dir: str):
    
    # authors
    df_authors = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORS_FN)
    df_authors['ID'] = range(1, len(df_authors) + 1)
    print(f'\n df_author: {df_authors.shape}  \n {df_authors.head(2)} \n')

    
    # # Extra data for e-index:

    # authorships
    df_authorships = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_AUTHORSHIPS_FN)
    print(f'\n df_authorships: {df_authorships.shape}  \n {df_authorships.head(5)} \n')

    # publications
    df_publications = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_FN)
    print(f'\n df_publications: {df_publications.shape}  \n {df_publications.head(5)} \n')

    # citations
    df_citations = io.read_file_from_tar_gz_as_dataframe(aps_os_data_tar_gz, constants.APS_OA_CITATIONS_FN)
    print(f'\n df_citations: {df_citations.shape}  \n {df_citations.head(5)} \n')




    ids = df_citations.id_publication_cited.unique()
    print(f"--- df_authorships with id_publication_cited: {df_authorships.query('id_publication in @ids').shape}")
    print(f"--- df_authorships WITHOUT id_publication_cited: {df_authorships.query('id_publication not in @ids').shape}")

    ids = df_citations.id_publication_citing.unique()
    print(f"--- df_authorships with id_publication_citing: {df_authorships.query('id_publication in @ids').shape}")
    print(f"--- df_authorships WITHOUT id_publication_citing: {df_authorships.query('id_publication not in @ids').shape}")

    ids = df_authorships.id_publication.unique()
    print(f"--- df_citations with id_publication_cited: {df_citations.query('id_publication_cited in @ids').shape}")
    print(f"--- df_citations WITHOUT id_publication_cited: {df_citations.query('id_publication_cited not in @ids').shape}")

    print(f"--- df_citations with id_publication_citing: {df_citations.query('id_publication_citing in @ids').shape}")
    print(f"--- df_citations WITHOUT id_publication_citing: {df_citations.query('id_publication_citing not in @ids').shape}")




    # merging
    df_citations = df_citations.merge(df_authorships[['id_publication','id_author']], left_on='id_publication_cited', right_on='id_publication', how='left').drop(columns=['id_publication'])
    print(f'\n df_citations (mwerge authors): {df_citations.shape}  \n {df_citations.head(2)} \n')
    df_citations = df_citations.merge(df_publications[['id_publication','publication_date']], left_on='id_publication_cited', right_on='id_publication', how='left')
    print(f'\n df_citations (mwerge publications): {df_citations.shape}  \n {df_citations.head(2)} \n')

    # merging
    df_authorships = df_authorships.merge(df_publications[['id_publication','publication_date']], on='id_publication', how='left')
    df_authorships['year'] = df_authorships['publication_date'].str.split('-').str[0].astype(int)
    print(f'\n df_authorships (merge publications): {df_authorships.shape}  \n {df_authorships.head(2)} \n')

    # time
    df_time = df_authorships.groupby('id_author').year.agg(['min','max']).reset_index().rename(columns={'min':'min_year','max':'max_year'})
    df_time.loc[:,'career_age'] = df_time.apply(lambda row: (row.max_year - row.min_year)+1, axis=1)
    print(f'\n df_time: {df_time.shape}  \n {df_time.head(2)} \n')



    # compute e-index
    df_e_index = df_citations.groupby('id_author').apply(lambda group: scholar.compute_e_index(group, col_publication='id_publication_cited', col_citation_from='id_publication_citing')).reset_index(name='e_index')
    df_authors = df_authors.merge(df_e_index[['id_author','e_index']], on='id_author', how='left')
    print(f'\n df_author (e_index): {df_authors.shape}  \n {df_authors.head(2)} \n')


    # compute normalized citation/publication age
    df_authors = df_authors.merge(df_time[['id_author','career_age','max_year','min_year']], on='id_author', how='left')
    df_authors.loc[:,'citations_per_paper_age'] = df_authors.apply(lambda row: 0 if row.works_count == 0 or row.career_age == 0 else (row.cited_by_count / row.works_count) / row.career_age, axis=1)
    print(f'\n df_author (citations_per_paper_age): {df_authors.shape}  \n {df_authors.head(2)} \n')



    # compute rankings
    for mid, (metric, col_name) in enumerate(constants.RANKING_METRICS.items()):
        
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



    # SAVE author stats as CSV
    fn = io.path_join(output_dir, constants.APS_OA_AUTHORS_STATS_FN)
    io.save_csv(df_authors, fn, index=False)


    # Process final output
    df_authors.sort_values('ID', ascending=True, inplace=True)
    data = [get_object(row) for index, row in tqdm(df_authors.iterrows(), total=len(df_authors), desc="Processing rows")]

    # Save as list of dictionaries
    fn = io.path_join(output_dir, constants.AUTHORS_RANKINGS_FN)
    io.save_dicts_to_text_file(dict_list=data, file_path=fn)



def get_object(row):
    # id_author,created_date,updated_date,name,orcid,two_year_mean_citedness,h_index,i10_index,works_count,cited_by_count

    # {"id_author_ranking": "ar_0000001", "id_author": "a00466979", 
    # "rr1_rank_citations": 7270, "rr1_rank_citations_percentile": 98.45, 
    # "rr2_rank_publications": 13, "rr2_rank_publications_percentile": 100.0, 
    # "rr3_rank_h_index": 18119, "rr3_rank_h_index_percentile": 96.13, 
    # "rr4_rank_e_index": 64635, "rr4_rank_e_index_percentile": 86.19, 
    # "rr5_rank_citation_publication_age": 409138, "rr5_rank_citation_publication_age_percentile": 12.59}

    # {"id_author_ranking": "ar_0289251", "id_author": "a00466972", 
    # "rr1_rank_citations": 244605, "rr1_rank_citations_percentile": 47.74, 
    # "rr2_rank_publications": 362244, "rr2_rank_publications_percentile": 22.61, 
    # "rr3_rank_h_index": 312969, "rr3_rank_h_index_percentile": 33.14, 
    # "rr4_rank_e_index": 350715, "rr4_rank_e_index_percentile": 25.08, 
    # "rr5_rank_citation_publication_age": 53960, "rr5_rank_citation_publication_age_percentile": 88.47}

    obj = {}
    obj['id_author_ranking'] = row.ID
    obj['id_author'] = row.ID
    obj['openalex_id'] = f"A{row.id_author}"
    
    for mid, (metric, col_name) in enumerate(constants.RANKING_METRICS.items()):
        
        col_rank = f"rr{mid+1}_rank_{metric}"
        col_perc = f"rr{mid+1}_rank_{metric}_percentile"

        obj[col_rank] = row[col_rank]
        obj[col_perc] = row[col_perc]

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