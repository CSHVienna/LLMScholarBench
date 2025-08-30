
from operator import index
from libs import io
from libs import constants
from libs import responses

class FactualityCheck(object):

    def __init__(self, aps_os_data_tar_gz: str|None, valid_responses_dir: str|None, model: str, task: str|None, max_workers: int, output_dir: str):
        """
        Initialize the FactualityCheck object
        """
        self.aps_os_data_tar_gz = aps_os_data_tar_gz
        self.valid_responses_dir = valid_responses_dir
        self.model = model
        self.task_name = task
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.fn_valid_with_fact_check = None
        self.df_valid_responses = None
        self.df_authors_metadata = None
        self.df_authors_mapping = None
        self._columns_to_hide = []
        self.set_filename()

    def set_filename(self):
        """
        Set the filename
        """
        fn = f"{self.model}.csv" if self.task_name is None else f"{self.model}_{self.task_name}.csv"
        self.fn_valid_with_fact_check = io.path_join(self.output_dir, constants.EXPERIMENT_AUDIT_FACTUALITY, fn)

    def already_checked(self):
        """
        Check if the responses have already been checked
        """
        return io.exists(self.fn_valid_with_fact_check)

    def load_valid_responses_with_factuality_check(self):
        """
        Load the valid responses with the factuality check
        """
        self.df_valid_responses = io.read_csv(self.fn_valid_with_fact_check, index_col=0)

    def save_valid_responses_with_factuality_check(self):
        """
        Save the valid responses with the factuality check
        """
        if constants.FACTUALITY_COLUMN_ID in self.df_valid_responses.columns:
            self.df_valid_responses.set_index(constants.FACTUALITY_COLUMN_ID, inplace=True)

        io.validate_path(self.fn_valid_with_fact_check)  
        io.save_csv(self.df_valid_responses.drop(columns=self._columns_to_hide), self.fn_valid_with_fact_check, index=True)

    def set_valid_responses(self, df_valid_responses):
        self.df_valid_responses = df_valid_responses
        if self.task_name is not None and self.task_name != constants.FACTUALITY_AUTHOR:
            self.df_valid_responses = self.df_valid_responses.query("task_name == @self.task_name").copy()
            self.df_valid_responses.reset_index(inplace=True)
            self.df_valid_responses.rename(columns={'index':constants.FACTUALITY_COLUMN_ID}, inplace=True)
            io.printf(f"Model: {self.model} - Task: {self.task_name} - {self.df_valid_responses.shape}")

    def load_valid_responses(self):
        """ 
        Load responses from the valid_responses_dir
        """
        self.df_valid_responses = responses.read_responses(self.valid_responses_dir, self.model, self.task_name)

    def run_factuality_check(self):
        """
        Run the factuality check
        """
        return
    
    def _assign_ids(self, column_names):
        unique_values = self.df_valid_responses[column_names].unique()
        df_unique_values = io.pd.DataFrame(sorted(unique_values), columns=[column_names], index=range(1, len(unique_values)+1))
        return df_unique_values

    def _load_author_metadata(self):
        """
        Load author metadata
        """
        self.df_authors_metadata = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_AUTHORS_DEMOGRAPHICS_FN)
        self.df_authors_metadata.rename(columns={'id_author':"id_author_oa"}, inplace=True)
        
        self.df_authors_mapping = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_AUTHORS_MAPPING_FN)

        self.df_authors_stats = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_AUTHORS_STATS_FN)
        self.df_authors_stats.rename(columns={'id_author':"id_author_oa"}, inplace=True)
        
        
    def _load_publications(self):
        """
        Load publications
        """
        self.df_publications = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_FN)
        self.df_publications.rename(columns={'id_publication':"id_publication_oa"}, inplace=True)

        self.df_publications_mapping = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_PUBLICATIONS_MAPPING_FN)
        self.df_publications_mapping.rename(columns={'oa_id':"id_publication_oa", "aps_id":'id_publication_aps'}, inplace=True)
        
        self.df_authorships = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_AUTHORSHIPS_FN)
        self.df_authorships.rename(columns={'id_publication':"id_publication_oa", "id_author":'id_author_oa'}, inplace=True)

    def _load_publication_topics(self):
        self.df_publication_topics = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_PUBLICTION_TOPICS)
        self.df_publication_topics.rename(columns={'id_publication':"id_publication_oa", 'id_topic':'id_topic_oa'}, inplace=True)
        self.df_topics = io.read_file_from_tar_gz_as_dataframe(self.aps_os_data_tar_gz, constants.APS_OA_TOPICS_FN)
        self.df_topics.rename(columns={'id_topic':"id_topic_oa"}, inplace=True)

    def _get_factual_mapping(self, df_unique_values, *args):    
        return
        
    @staticmethod
    def get_minority_baselines(attribute, df_gt_stats, df_all_authors_demographics, df_all_authors_stats):
        from libs.factuality import author

        minority = constants.GENDER_FEMALE if attribute == 'gender' else constants.ETHNICITY_BLACK

        # Field
        baseline_field = df_gt_stats.query("label in ['Condensed Matter & Materials Physics', 'Physics Education Research']")[f'{attribute}_fractions'][minority]
        baseline_field.rename(index={'Condensed Matter & Materials Physics':'CMMP', 'Physics Education Research':'PER'}, inplace=True)

        # Epoch
        df_periods = df_all_authors_stats.set_index('id_author_oa').join(df_all_authors_demographics[['id_author','gender', 'ethnicity']].set_index('id_author'))
        df_periods = df_periods[['min_year', 'max_year', 'career_age', 'gender', 'ethnicity']]
        decades = [1950, 2000]
        fvalues = []
        for decade in decades:
            k = f"s{decade}s"
            df_periods.loc[:,k] = df_periods.apply(lambda row: not ((row.max_year < decade) or (row.min_year > decade + 10)) , axis=1)
            tmp = df_periods.query(f"{k} == True")
            f = tmp.query(f"{attribute}=='{minority}'").shape[0] / tmp.shape[0]
            fvalues.append(f)
        baseline_epoch = io.pd.Series(fvalues, index=decades)

        # Seniority
        df_periods.loc[:, 'seniority'] = df_periods.career_age.apply(lambda v: author.get_seniority(v))
        counts = df_periods.groupby(['seniority', attribute]).size().reset_index(name='count')
        counts['fraction'] = counts['count'] / counts.groupby('seniority')['count'].transform('sum')
        result = counts.pivot(index='seniority', columns=attribute, values='fraction')
        baseline_seniority = result.loc[constants.TASK_SENIORITY_PARAMS, minority]

        return {constants.EXPERIMENT_TASK_FIELD: baseline_field, 
                constants.EXPERIMENT_TASK_EPOCH: baseline_epoch,
                constants.EXPERIMENT_TASK_SENIORITY: baseline_seniority}