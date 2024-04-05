CONFIG_FILE=${1:-configs/meqsum_gpt35_shot0_azure.yaml}

python run.py --eval_file data/meqsum-test_clean.json --result_dir results --dataset_and_tag meqsum \
    --config_file $CONFIG_FILE
