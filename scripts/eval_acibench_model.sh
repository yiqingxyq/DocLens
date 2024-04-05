CACHE_DIR="llama_models"

MODEL=gpt-35-turbo
SHOT=1
TAG="-persection" # TAG="" for full-note generation
DEFAULT_SAVENAME=acibench-test1${TAG}-${MODEL}-shot${SHOT}-azure-quick_test3 # remove -quick_test3 if no quick test

SAVENAME=${1:-${DEFAULT_SAVENAME}} 
EVAL_MODEL=${2:-Mistral}

# generate reference subclaims
python data_processing/divide_section.py --result_file data/ACI-Bench-TestSet-1_clean.json

python claim_evaluation/generate_subclaims.py --eval_file data/ACI-Bench-TestSet-1_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_persection_subclaim_generation.json --azure

# generate output subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/ACI-Bench-TestSet-1_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_persection_subclaim_generation.json --azure

# claim recall
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name acibench --mode claim_recall --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_claim_entail_Mistral.json --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

# claim precision 
python claim_evaluation/run_entailment_model.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name acibench --mode claim_precision --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_claim_entail_Mistral.json --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

# eval citations 
python citation_evaluation/eval_citation_model.py --result_file results/${SAVENAME}.json \
    --dataset_name acibench --split_method sent --get_persection_score \
    --prompt_file citation_evaluation/prompts/acibench_citation_entail_Mistral.json \
    --eval_model $EVAL_MODEL --cache_dir $CACHE_DIR

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name acibench \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model $EVAL_MODEL