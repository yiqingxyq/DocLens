CACHE_DIR="models"

SAVENAME=$1
EVAL_MODEL=${2:-Mistral}
PROMPT_FILE=${3:-"citation_evaluation/prompts/general_citation_entail_Mistral.json"}
DATASET_NAME=${4:-user_dataset}

# eval citations 
python citation_evaluation/eval_citation_model.py --result_file results/${SAVENAME}.json \
    --dataset_name $DATASET_NAME --split_method citation \
    --prompt_file $PROMPT_FILE \
    --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_citations --eval_model $EVAL_MODEL