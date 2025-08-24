import pandas as pd

from libs import text
from libs import constants
from libs.factuality.factcheck import FactualityCheck

class FactualityEpoch(FactualityCheck):

    def __init__(self, aps_os_data_tar_gz: str, valid_responses_dir: str|None, model: str, max_workers: int, output_dir: str):
        super().__init__(aps_os_data_tar_gz, valid_responses_dir, model, constants.EXPERIMENT_TASK_EPOCH, max_workers, output_dir)
        self.column_epoch = 'years'
        self.column_task_param = 'task_param'
        self._columns_to_hide = constants.FACTUALITY_AUTHOR_METADATA_TO_HIDE + constants.FACTUALITY_EPOCH_TO_HIDE

    def run_factuality_check(self):
        # Run factuality check
        super().run_factuality_check()

        # 2. Check years with years of activity (df_valid_responses already has the author metadata)
        new_cols = ['fact_epoch_requested', 'fact_epoch_llm_in_gt', 'fact_epoch_gt_in_llm', 'fact_epoch_overlap', 'fact_epoch_intersection_length', 'fact_epoch_jaccard_sim']
        self.df_valid_responses[new_cols] = self.df_valid_responses.apply(self._process_row, axis=1)
        
    def _process_row(self, row):
        llm1, lmm2 = row[self.column_epoch].split("-")
        return get_overlap_range_metrics(int(row[self.column_task_param].replace('s','')), 
                                         int(llm1), int(lmm2), 
                                         row['year_first_publication'], row['year_last_publication'])

def get_overlap_range_metrics(requested_epoch, start1, end1, start2, end2):
    # 1 llm: start1, end1: start and end year according to LLM response
    # 2 gt : start2, end2: start and end year according to APS GT.

    # is_requested_epoch (the author is active in the requested epoch) - no llm answer
    # is_requested_epoch = requested_epoch >= start2 and requested_epoch <= end2 # OLD
    # If either year is NaN, return False; otherwise check overlap
    is_requested_epoch = pd.notna(start2) and pd.notna(end2) and ~(end2 < requested_epoch or start2 > requested_epoch + 10)
    
    # Containment
    r1_in_r2 = start1 >= start2 and end1 <= end2 # llm answer is within gt
    r2_in_r1 = start2 >= start1 and end2 <= end1 # llm answer is otuside gt

    # Overlap
    overlap = start1 <= end2 and start2 <= end1 # the two ranges (llm and gt) overlap

    # Intersection Length
    intersection_length = max(0, min(end1, end2) - max(start1, start2))

    # Union Length
    union_length = max(end1, end2) - min(start1, start2)

    # Jaccard Similarity
    jaccard_similarity = intersection_length / union_length if union_length > 0 else 0

    return pd.Series([is_requested_epoch, r1_in_r2, r2_in_r1, overlap, intersection_length, jaccard_similarity])
