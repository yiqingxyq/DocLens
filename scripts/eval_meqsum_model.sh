CACHE_DIR="models"

MODEL=gpt-35-turbo
SHOT=2
DEFAULT_SAVENAME=meqsum-${MODEL}-shot${SHOT}-azure-quick_test3

SAVENAME=${1:-${DEFAULT_SAVENAME}} 
EVAL_MODEL=${2:-Mistral}

# claim recall
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --mode claim_recall \
    --prompt_file claim_evaluation/prompts/meqsum_claim_entail_Mistral.json --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

# claim precision 
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --mode claim_precision \
    --prompt_file claim_evaluation/prompts/meqsum_claim_entail_Mistral.json --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

# claim same 
if [ "$EVAL_MODEL" != "TRUE" ]
then
    python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.json \
        --dataset_name meqsum --mode same \
        --prompt_file claim_evaluation/prompts/meqsum_claim_entail_same_Mistral.json --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR
fi

# eval citations 
python citation_evaluation/eval_citation_model.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --split_method sent \
    --prompt_file citation_evaluation/prompts/meqsum_citation_entail_Mistral.json \
    --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name meqsum \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model $EVAL_MODEL