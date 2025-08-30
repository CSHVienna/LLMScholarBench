from libs import constants
from libs.factuality.factcheck import FactualityCheck

class FactualityTopK(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.EXPERIMENT_TASK_TOPK, max_workers, output_dir)
    
    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()
        
        
        return
