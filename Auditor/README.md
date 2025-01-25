# Scripts

## Pipeline

1. Purge all responses to label validity (valid, verbose, fixed, invalid, quota, error)

        `python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model gemma2-9b`

2. Run factuality checks

    2.1 Start with factuality author

        `python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b`

    2.2 Run factuality checks for tasks: `field`, `epoch`, `seniority` (order doesn't matter)

        `python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b --task_name field`

3. Run metadata 

    3.1 Run statistics for gender and ethnicity representation by discipline (GT APS-OA)

        `python preprocessing/batch_disciplines.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --aps_data_dir ../data/aps_20240130 --output ../results`


3. Plots

    3.1 Validity

        `notebooks/1_basic_stats.ipynb`

    3.2 Consistency

        `notebooks/2_consistency.ipynb`

    3.3 Factuality

        `notebooks/3_factuality.ipynb`

    3.4 Demographics

        `notebooks/4_demographics.ipynb`

    3.5 Prestige

        `notebooks/5_prestige.ipynb`

    