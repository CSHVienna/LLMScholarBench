import pandas as pd
from pyparsing import col

from libs import text
from libs import constants
from libs.factuality.factcheck import FactualityCheck

class FactualityField(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str|None, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.EXPERIMENT_TASK_FIELD, max_workers, output_dir)
        self.column_doi = 'doi'
        self.column_name = 'name'
        self._columns_to_hide = constants.FACTUALITY_AUTHOR_METADATA_TO_HIDE + constants.FACTUALITY_FIELD_TO_HIDE

    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()

        # 1. check if DOI exists
        self._factuality_doi()

        # 2. check if DOI is from author, and if Field from author is correct
        self._factuality_doi_author_field()


    def _factuality_doi(self):
        # 2. unique DOIs
        df_unique_dois = self._assign_ids(self.column_doi)

        # 3. check if DOI exists
        df_unique_dois = self._get_factual_mapping(df_unique_dois, constants.FACTUALITY_AUTHOR_FIELD_DOI)

        # 4. merge the factual mapping with the responses (exists in OpenAlex)
        self.df_valid_responses = self.df_valid_responses.merge(df_unique_dois, on=self.column_doi, how='left')

        # 5. merge with APS data (exists in APS)
        self.df_valid_responses = self.df_valid_responses.merge(self.df_publications_mapping[['id_publication_oa','id_publication_aps']], on='id_publication_oa', how='left')


    def _get_factual_mapping(self, df_unique_dois, threshold=1):
        super()._get_factual_mapping(df_unique_dois)

        # load publications
        super()._load_publications()

        # LLM responses
        dfA = df_unique_dois.copy()
        dfA.loc[:,self.column_doi] = dfA.loc[:,self.column_doi].str.lower()

        # Author metadata (GT)
        dfB = self.df_publications[['id_publication_oa',"doi"]].set_index('id_publication_oa').copy()
        dfB.loc[:,'doi'] = dfB.doi.str.lower()

        # Setting up record linkage
        column_block = 'doi'
        column_pairs = [("doi", "doi", "jarowinkler", 0.99, "doi")]
        valid_matches = text.find_matching_texts(dfA, dfB, column_block=column_block, column_pairs_to_evaluate=column_pairs, threshold=threshold)

        # adding new information (mapping)
        df_unique_dois = df_unique_dois.join(valid_matches[['id_publication_oa','total_matches']], how='left')
        df_unique_dois.rename(columns={"total_matches": "fact_doi_score"}, inplace=True)
        return df_unique_dois
    
    def _factuality_doi_author_field(self):
        # load topics
        super()._load_publication_topics()

        # merge info author topic
        tmp = self.df_publication_topics.merge(self.df_topics[['id_topic_oa','domain_name','field_name','subfield_name']], on='id_topic_oa', how='left').copy()
        tmp = tmp[['id_publication_oa','domain_name','field_name','subfield_name']]
        tmp = tmp.merge(self.df_authorships[['id_publication_oa','id_author_oa']], on='id_publication_oa')
        

        cols = ['id_author_oa','id_publication_oa','domain_name','field_name','subfield_name']
        tmp = tmp[cols].groupby(['id_author_oa','id_publication_oa'], as_index=False).agg({'domain_name': lambda x: list(set(x)),
                                                                                            'field_name': lambda x: list(set(x)),
                                                                                            'subfield_name': lambda x: list(set(x))})
        tmp.rename(columns={'domain_name':'domain_name_list',
                            'field_name':'field_name_list',
                            'subfield_name':'subfield_name_list'}, inplace=True)


        # @NOTE: The following hasn't been run yet (2025-02-02)

        # Check the topics of publications
        self.df_valid_responses = self.df_valid_responses.merge(tmp[['id_publication_oa','subfield_name_list']], on=['id_publication_oa'], how='left')
        self.df_valid_responses.loc[:,'fact_doi_field'] = self.df_valid_responses.apply(lambda row: evaluate_factuality_field(row) , axis=1)
        self.df_valid_responses.drop(columns=['subfield_name_list'], inplace=True)

        # Check the topics of authors
        self.df_valid_responses = self.df_valid_responses.merge(tmp[['id_author_oa','subfield_name_list']], on=['id_author_oa'], how='left')
        self.df_valid_responses.loc[:,'fact_author_field'] = self.df_valid_responses.apply(lambda row: evaluate_factuality_field(row) , axis=1)
        self.df_valid_responses.drop(columns=['subfield_name_list'], inplace=True)

        # Check the topics of publications and authors
        self.df_valid_responses = self.df_valid_responses.merge(tmp[['id_publication_oa', 'id_author_oa','subfield_name_list']], on=['id_publication_oa','id_author_oa'], how='left')
        self.df_valid_responses.rename(columns={'id_author_oa':'fact_author'}, inplace=True)
        self.df_valid_responses.loc[:,'fact_doi_author_field'] = self.df_valid_responses.apply(lambda row: evaluate_factuality_field(row) , axis=1)

        
def evaluate_factuality_field(row):
    if isinstance(row['subfield_name_list'], list):
        # Check if any element in the list is NaN
        if any(pd.isna(x) for x in row['subfield_name_list']):
            # Handle case where there's a NaN in the list
            return None
    elif pd.isna(row['subfield_name_list']):
        return None

    for c in row['subfield_name_list']:
        if constants.FACTUALITY_FIELD_PER_SF in c and row['task_param'] == constants.FACTUALITY_FIELD_PER:
            return True
        elif constants.FACTUALITY_FIELD_CMP_SF in c and row['task_param'] == constants.FACTUALITY_FIELD_CMP:
            return True
        
    return False