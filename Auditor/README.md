# Scripts

## Pipeline

1. Go to `code` and set up the library enviroment path. All batch code will be run from this directory.

        export PYTHONPATH="${PYTHONPATH}:."

2. Purge all responses to label validity (valid, verbose, fixed, invalid, quota, error)

        python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model gemma2-9b

    Or using parallel to execeute all models' outputs:

        parallel -j 6 python preprocessing/batch_valid_answers.py --experiments_dir ../data/experiments --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b

    Note: Remember to pass the `boolean` parameter `--temperature_analysis` if the results contain multiple temperature values per model.

3. Run factuality checks


    3.1 Start with factuality author (without passing the ``--task_param`` param)

        python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b


    Or using parallel to execeute all models' outputs:

        parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b


    3.2 Run factuality checks for tasks: `field`, `epoch`, `seniority` (order doesn't matter)

        python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model gemma2-9b --task_name field

    Or using parallel to execeute all models' outputs:

        parallel -j 6 python preprocessing/batch_factuality.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --max_workers 10 --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: field epoch seniority


4. Run metadata 


    4.1 Run statistics for gender and ethnicity representation by discipline (GT APS-OA)

        python preprocessing/batch_disciplines.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --aps_data_zip ../data/aps_20240130_2e52fdd7260ea462878821948a2a463ed9acb58a.zip --output ../results


    4.2 Run statistics for similarity using co-authorship networks, and metadata (demographics, scholarly metrics, and affiliations)

        python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model gemma2-9b --task_name top_k

        
    Or using parallel to execeute all models' outputs:

        
        parallel -j 6 python preprocessing/batch_similarities.py --aps_os_data_tar_gz ../data/final_dataset.tar.gz --valid_responses_dir ../results/valid_responses --output_dir ../results --model {1} --task_name {2} ::: gemma2-9b llama-3.1-8b llama-3.1-70b llama3-8b llama3-70b mixtral-8x7b ::: top_k field epoch seniority twins


5. Plots

    3.1 Validity

        notebooks/1_basic_stats.ipynb

    3.2 Consistency

        notebooks/2_consistency.ipynb

    3.3 Factuality

        notebooks/3_factuality.ipynb

    3.4 Demographics

        notebooks/4_demographics.ipynb

    3.5 Prestige

        notebooks/5_prestige.ipynb

    3.6 Similarity

        notebooks/6_similarity.ipynb

    