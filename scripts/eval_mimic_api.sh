MODEL=gpt-35-turbo
SHOT=6
DEFAULT_SAVENAME=mimic-sample200-${MODEL}-shot${SHOT}-azure-quick_test3

SAVENAME=${1:-${DEFAULT_SAVENAME}} 

# generate reference subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json --azure

# generate output subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json --azure

# claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name mimic --mode claim_recall \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json --azure 

# claim precision 
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name mimic --mode claim_precision \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json --azure 

# eval citations 
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name mimic --split_method citation \
    --prompt_file citation_evaluation/prompts/mimic_citation_entail.json --azure

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name mimic \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model GPT