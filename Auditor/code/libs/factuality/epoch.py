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
        print(self.df_valid_responses['fact_epoch_requested'].value_counts())

    def _process_row(self, row):

        val = row[self.column_epoch].replace('–','-').lower()

        if 'not active' in val:
            llm1 = None
            llm2 = None
        elif '-' in val:
            llm1, llm2 = val.split("-")[0], val.split("-")[1]

            if 'present' in llm1:
                llm1 = '2025'
            if 'present' in llm2:
                llm2 = '2025'
            if 'yyyy' in llm1:
                llm1 = ''
            if 'yyyy' in llm2:
                llm2 = ''

            llm1 = int(llm1.replace('_','').replace('s','').split(':')[0]) if llm1 not in constants.NONE else None
            llm2 = int(llm2.replace('_','').replace('s','').split(':')[0]) if llm2 not in constants.NONE else llm1 + 10 if llm1 else None
        elif '-' not in val:
            try:
                llm1 = int(val)
                llm2 = llm1 + 10
            except:
                llm1 = None
                llm2 = None

        return get_overlap_range_metrics(int(row[self.column_task_param].replace('s','')), 
                                         int(llm1) if llm1 else None, 
                                         int(llm2) if llm2 else None, 
                                         int(row['year_first_publication']) if pd.notna(row['year_first_publication']) else None,
                                         int(row['year_last_publication']) if pd.notna(row['year_last_publication']) else None)

def get_overlap_range_metrics(requested_epoch, start_llm, end_llm, start_gt, end_gt):
    # 1 llm: start1, end1: start and end year according to LLM response
    # 2 gt : start2, end2: start and end year according to APS GT.

    # is_requested_epoch (the author is active in the requested epoch) - no llm answer
    # is_requested_epoch = requested_epoch >= start2 and requested_epoch <= end2 # OLD
    # If either year is NaN, return False; otherwise check overlap
    available_gt = (pd.notna(start_gt) and pd.notna(end_gt))
    if not available_gt:
        is_requested_epoch = None
        real_overlap = None
    else:
        real_overlap = not (end_gt < requested_epoch or start_gt > (requested_epoch + 10))
        is_requested_epoch = available_gt and real_overlap

    #print(f"is_requested_epoch: {is_requested_epoch} | available_gt: {available_gt} | real_overlap: {real_overlap} | start_gt {start_gt} ({pd.notna(start_gt)}) |  end_gt: {end_gt} ({pd.notna(end_gt)}) | requested_epoch: {requested_epoch} | (requested_epoch + 10): {requested_epoch + 10}")
    

    if not available_gt or start_llm is None or end_llm is None:
        llm_in_gt = None
        gt_in_llm = None
        overlap = None
        intersection_length = None
        union_length = None
        jaccard_similarity = None

    else:
        # Containment
        llm_in_gt = start_llm >= start_gt and end_llm <= end_gt # llm answer is within gt
        gt_in_llm = start_gt >= start_llm and end_gt <= end_llm # llm answer is otuside gt (gt in llm)

        # Overlap
        # overlap = start_llm <= end_gt and start_gt <= end_llm # the two ranges (llm and gt) overlap # old
        overlap = max(start_llm, start_gt) <= min(end_llm, end_gt)

        # Intersection Length
        intersection_length = max(0, min(end_llm, end_gt) - max(start_llm, start_gt))

        # Union Length
        union_length = max(end_llm, end_gt) - min(start_llm, start_gt)

        # Jaccard Similarity
        jaccard_similarity = intersection_length / union_length if union_length > 0 else 0

    return pd.Series([is_requested_epoch, llm_in_gt, gt_in_llm, overlap, intersection_length, jaccard_similarity])
