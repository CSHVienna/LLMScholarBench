from libs import constants
from libs.factuality.factcheck import FactualityCheck

class FactualityTwins(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.EXPERIMENT_TASK_TWINS, max_workers, output_dir)

    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()
        
        # 1. load valid responses
        self.load_valid_responses()

        # 2. get unique names in responses, and assign an ID
        unique_names = self.df_valid_responses["name"].unique().reset_index().rename(columns={"index":"name_id"})
        print(unique_names)
        print(len(unique_names))
        
        return
