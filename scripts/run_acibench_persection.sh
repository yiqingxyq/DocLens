CONFIG_FILE=${1:-configs/aci_gpt4_shot2_azure.yaml}
 
for section in subjective objective_exam objective_results assessment_and_plan
do 
    python run.py --prompt_file prompts/acibench_${section}_annotated-citation.json \
        --eval_file data/ACI-Bench-TestSet-1_clean.json --result_dir results --dataset_and_tag acibench-test1-${section} \
        --config_file $CONFIG_FILE
done 
