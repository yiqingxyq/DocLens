SAVENAME=$1
PROMPT_FILE=${2:-"citation_evaluation/prompts/general_citation_entail.json"}
DATASET_NAME=${3:-user_dataset}

# eval citations 
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name $DATASET_NAME --split_method citation \
    --prompt_file $PROMPT_FILE --azure

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_citations --eval_model GPT