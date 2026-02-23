source init.sh

python ../code/preprocessing/batch_refusals.py --summaries_dirs ../results/temperature_analysis/summaries/ ../results/interventions/summaries/ --summaries_sources temperature interventions --output_dir ../results/