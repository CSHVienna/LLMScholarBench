import pandas as pd

from libs import text
from libs import constants
from libs.factuality.factcheck import FactualityCheck
from libs.factuality.author import get_seniority

class FactualitySeniority(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str|None, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.EXPERIMENT_TASK_SENIORITY, max_workers, output_dir)
        self.column_age = 'career_age'
        self.column_task_param = 'task_param'
        self._columns_to_hide = constants.FACTUALITY_AUTHOR_METADATA_TO_HIDE + constants.FACTUALITY_SENIORITY_TO_HIDE

    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()

        # 2. Check career_age
        self.df_valid_responses.loc[:,'seniority_llm'] = self.df_valid_responses.loc[:,'career_age'].apply(lambda x: get_seniority(x))
        self.df_valid_responses.loc[:,'fact_seniority_active'] = self.df_valid_responses.apply(lambda row: row.seniority_active == row.seniority_llm, axis=1)
        self.df_valid_responses.loc[:,'fact_seniority_now'] = self.df_valid_responses.apply(lambda row: row.seniority_now == row.seniority_llm, axis=1)
        
        # early_career, senior
        self.df_valid_responses.loc[:,'fact_seniority_active_requested'] = self.df_valid_responses.apply(lambda row: row.seniority_active == row[self.column_task_param], axis=1)
        self.df_valid_responses.loc[:,'fact_seniority_now_requested'] = self.df_valid_responses.apply(lambda row: row.seniority_now == row[self.column_task_param], axis=1)


        