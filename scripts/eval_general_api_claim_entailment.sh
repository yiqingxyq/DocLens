SAVENAME=$1
REFERENCE=$2
PROMPT_FILE=${3:-"claim_evaluation/prompts/general_claim_entail.json"}
DATASET_NAME=${4:-user_dataset}

# claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name $DATASET_NAME --mode claim_recall \
    --prompt_file $PROMPT_FILE --azure 

# claim precision 
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name $DATASET_NAME --mode claim_precision \
    --prompt_file $PROMPT_FILE --azure 

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name $DATASET_NAME \
    --eval_claim_recall --eval_claim_precision --eval_model GPT
