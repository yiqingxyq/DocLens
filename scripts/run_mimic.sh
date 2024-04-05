CONFIG_FILE=${1:-configs/mimic_gpt4_shot12_azure.yaml}

python run.py --eval_file data/mimic-sampled200_clean.json --result_dir results --dataset_and_tag mimic-sample200 \
    --config_file $CONFIG_FILE