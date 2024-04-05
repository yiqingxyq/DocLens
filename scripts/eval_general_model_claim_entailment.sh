CACHE_DIR="models"

SAVENAME=$1
REFERENCE=$2
EVAL_MODEL=${3:-Mistral}
PROMPT_FILE=${4:-"claim_evaluation/prompts/general_claim_entail_Mistral.json"}
DATASET_NAME=${5:-user_dataset}


# claim recall
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name $DATASET_NAME --mode claim_recall \
    --prompt_file $PROMPT_FILE --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

# claim precision 
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name $DATASET_NAME --mode claim_precision \
    --prompt_file $PROMPT_FILE --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_claim_recall --eval_claim_precision --eval_model $EVAL_MODEL