source init.sh

# validity temperature analysis 
nice -n 10 parallel -j 10 python ../code/preprocessing/batch_valid_answers.py --experiments_dir ../data/data-temperature --max_workers 10 --temperature_analysis --output_dir ../results/temperature_analysis --model {1} ::: llama-3.1-8b llama-4-scout llama-4-mav gpt-oss-20b gpt-oss-120b qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b-2507 qwen3-235b-a22b-2507 gemma-3-12b gemma-3-27b mistral-small-3.2-24b mistral-medium-3 llama-3.1-70b llama-3.3-70b llama-3.1-405b grok-4-fast deepseek-chat-v3.1 deepseek-r1-0528 gemini-2.5-flash gemini-2.5-flash-grounded gemini-2.5-pro gemini-2.5-pro-grounded

# validity interventions open-weight
nice -n 10 parallel -j 10 python ../code/preprocessing/batch_valid_answers.py --experiments_dir ../data/data-openrouter --max_workers 10 --output_dir ../results/interventions --model {1} ::: llama-3.1-8b llama-4-scout llama-4-mav gpt-oss-20b gpt-oss-120b qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b-2507 qwen3-235b-a22b-2507 gemma-3-12b gemma-3-27b mistral-small-3.2-24b mistral-medium-3 llama-3.1-70b llama-3.3-70b llama-3.1-405b grok-4-fast deepseek-chat-v3.1 deepseek-r1-0528

# validity interventions gemini
nice -n 10 parallel -j 10 python ../code/preprocessing/batch_valid_answers.py --experiments_dir ../data/data-gemini --temperature_analysis --max_workers 10 --output_dir ../results/interventions --model {1} ::: gemini-2.5-flash gemini-2.5-flash-grounded gemini-2.5-pro gemini-2.5-pro-grounded
