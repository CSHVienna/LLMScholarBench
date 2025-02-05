import pandas as pd
import numpy as np

from libs.factuality.factcheck import FactualityCheck
from libs import text
from libs import constants

class FactualityAuthor(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.FACTUALITY_AUTHOR, max_workers, output_dir)
        self.column_name = 'clean_name'

    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()

        # 1. load responses
        self.load_valid_responses()

        # 2. get unique names in responses, and assign an ID
        df_unique_names = self._assign_ids(self.column_name)

        # 3. look for the names in the APS data (found in APS? found in OpenAlex?)
        df_unique_names = self._get_factual_mapping(df_unique_names, threshold=constants.FACTUALITY_AUTHOR_THRESHOLD)

        # 4. merge the factual mapping with the responses (exists in OpenAlex)
        self.df_valid_responses = self.df_valid_responses.merge(df_unique_names, on=self.column_name, how='left')

        # 5. merge with APS data (exists in APS)
        df_authors_mapping = self.df_authors_mapping[['id_author_oa','id_author_aps']].groupby(['id_author_oa'], as_index=False).agg({'id_author_aps': list})
        df_authors_mapping.rename(columns={'id_author_aps':'id_author_aps_list'}, inplace=True)

        self.df_valid_responses = self.df_valid_responses.merge(df_authors_mapping, on='id_author_oa', how='left')

        # 6. append demographics and stats
        self.df_valid_responses = self.df_valid_responses.merge(self.df_authors_metadata[['id_author_oa']+constants.FACTUALITY_AUTHOR_DEMOGRAPHICS_TO_HIDE], on='id_author_oa', how='left')
        self.df_valid_responses = self.df_valid_responses.merge(self.df_authors_stats[['id_author_oa']+constants.FACTUALITY_AUTHOR_STATS_TO_HIDE], on='id_author_oa', how='left')

        # 7. more stats
        self._load_publications()
        df_authorship_publications = self.df_authorships.merge(self.df_publications[['id_publication_oa','publication_date']], on='id_publication_oa', how='left')
        df_authorship_publications.loc[:,'year'] = df_authorship_publications.publication_date.apply(lambda x: int(x[:4]))
        df_authorship_publications = df_authorship_publications.groupby('id_author_oa').year.agg(['min','max']).reset_index().rename(columns={'min':'year_first_publication','max':'year_last_publication'})
        df_authorship_publications.loc[:,'academic_age'] = df_authorship_publications.apply(lambda row: (row.year_last_publication-row.year_first_publication) + 1, axis=1)
        df_authorship_publications.loc[:,'age_now'] = df_authorship_publications.apply(lambda row: (constants.FACTUALITY_AUTHOR_YEAR_NOW-row.year_first_publication) + 1, axis=1)
        df_authorship_publications.loc[:,'seniority_active'] = df_authorship_publications.loc[:,'academic_age'].apply(lambda x: get_seniority(x))
        df_authorship_publications.loc[:,'seniority_now'] = df_authorship_publications.loc[:,'age_now'].apply(lambda x: get_seniority(x))

        self.df_valid_responses = self.df_valid_responses.merge(df_authorship_publications, on='id_author_oa', how='left')

    def _get_factual_mapping(self, df_unique_names, threshold=5):
        super()._get_factual_mapping(df_unique_names, threshold)

        # Load author metadata
        super()._load_author_metadata()

        # LLM responses
        dfA = df_unique_names.copy()
        dfA.rename(columns={self.column_name: "display_name"}, inplace=True)
        dfA.loc[:,'last_name'] = dfA.display_name.str.split().str[-1]
        dfA.loc[:,'first_name'] = dfA.display_name.str.split().str[0]
        dfA.loc[:,'second_name'] = dfA.display_name.apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
        dfA.loc[:,'first_initial'] = dfA.first_name.str[0]

        # Author metadata (GT)
        dfB = self.df_authors_metadata[['id_author_oa',"display_name","alternative_names","longest_name","last_name","first_name"]].set_index('id_author_oa').copy()
        dfB.loc[:,'display_name'] = dfB.loc[:,'display_name'].apply(lambda x: text.clean_name(x))
        dfB.loc[:,'second_name'] = dfB.longest_name.apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
        dfB.loc[:,'second_name'] = dfB.loc[:,'second_name'].apply(lambda x: text.clean_name(x))
        dfB.loc[:,'longest_name'] = dfB.loc[:,'longest_name'].apply(lambda x: text.clean_name(x))
        dfB.loc[:,'first_name'] = dfB.loc[:,'first_name'].apply(lambda x: text.clean_name(x))
        dfB.loc[:,'last_name'] = dfB.loc[:,'last_name'].apply(lambda x: text.clean_name(x))
        dfB.loc[:,'first_initial'] = dfB.first_name.str[0]

        # Setting up record linkage
        column_block = 'last_name'
        column_pairs = [("display_name", "display_name","jarowinkler",0.85,"display_name"),
                        ("display_name", "longest_name","jarowinkler",0.85,"d_longest_name"),
                        ("first_name", "first_name","jarowinkler",0.7,"first_name"),
                        ("last_name", "last_name","jarowinkler",0.7,"last_name"),
                        ("second_name", "second_name","jarowinkler",0.7,"second_name"),
                        ("first_initial", "first_initial","jarowinkler",0.7,"first_initial"),
                        ("display_name", "last_name","jarowinkler",0.7,"d_last_name"),
                        ("display_name", "first_name","jarowinkler",0.7,"d_first_name"),
                        ("display_name", "alternative_names","jarowinkler",0.7,"d_alternative_names")]
        valid_matches = text.find_matching_texts(dfA, dfB, column_block=column_block, column_pairs_to_evaluate=column_pairs, threshold=threshold)
        
        # adding new information (mapping)
        df_unique_names = df_unique_names.join(valid_matches[['id_author_oa','total_matches']], how='left')
        df_unique_names.rename(columns={"total_matches": "fact_author_score"}, inplace=True)
        df_unique_names.rename(columns={"display_name": self.column_name}, inplace=True)

        return df_unique_names


def get_seniority(val):

    if type(val) == str:
        if '+' in val:
            val = int(val.replace('+','').strip())
        elif 'years' in val:
            val = val.split('years')[0].strip()
            val = val.replace('Approximately','').replace('Over','').strip()

            if '-' in val:
                mi, ma = val.split('-')
                val = (int(mi) + int(ma)) / 2

            val = int(val)
        

    if type(val) == int:
        if val <= 10:
            return constants.FACTUALITY_SENIORITY_EARLY_CAREER
        elif val >= 20:
            return constants.FACTUALITY_SENIORITY_SENIOR_CAREER
        return constants.FACTUALITY_SENIORITY_MID_CAREER
