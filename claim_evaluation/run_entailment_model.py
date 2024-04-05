import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from vllm import LLM, SamplingParams

AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def _run_nli_autoais(passage, claim, autoais_model, autoais_tokenizer):
    passage_len = len(passage.split(' '))
    if passage_len > 500:
        print(f'Truncate passage from {passage_len} to 500')
        passage = ' '.join(passage.split(' ')[:500])
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=3)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if AUTOAIS_MODEL == 'google/t5_xxl_true_nli_mixture':
        inference = 1 if result == "1" else 0
    return inference


def call_llm(
        model, 
        prompt,
        temperature=0.7,
        top_p=0.95,
        n=1,
        max_new_tokens=128,
        min_new_tokens=1,
        max_at_once=None,
        stop=None,
        max_attempts=1,
    ):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n,
        stop=stop,
        include_stop_str_in_output=True,
    )
    
    outputs = model.generate(prompt, sampling_params, use_tqdm=False)
    texts = [x.text for output in outputs for x in output.outputs]
    
    return texts 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    
    # evaluation setting
    parser.add_argument("--mode", type=str, default="claim_recall", choices=['claim_recall','claim_precision','same'])
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    
    # evaluation model
    parser.add_argument('--prompt_file', default=None, help='filename of the prompt dict .json.')
    parser.add_argument("--eval_model", type=str, default="Mistral", choices=['TRUE', 'Mistral'])
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--cache_dir", type=str, default="llama_models")
    
    args = parser.parse_args()
    
    result_file, dataset_name, mode = args.result_file, args.dataset_name, args.mode
    prompt_file, eval_model, max_new_tokens, cache_dir = args.prompt_file, args.eval_model, args.max_new_tokens, args.cache_dir
    
    if mode == 'claim_recall':
        savefile = result_file.replace('.json', f'.claim_scores.{eval_model}')
    elif mode == 'claim_precision':
        savefile = result_file.replace('.json', f'.output_claim_scores.{eval_model}')
    elif mode == 'same':
        # For MeQSum, since the output only contains one line, we check:
        # (1) whether the output question is a specification of the reference question
        # (2) whether the reference question is a specification of the output question
        # (3) whether the two questions are essentially the same
        assert dataset_name == 'meqsum'
        savefile = result_file.replace('.json', f'.same_scores.{eval_model}')
        
    if not args.use_persection_claims:
        SECTION_DIVISIONS = ['full']
        
    if eval_model == 'Mistral':
        assert prompt_file is not None 
    
    output_data = json.load(open(result_file, 'r')) # a list of dicts
    
    print( f"Saving scores to {savefile.split('/')[-1]}..") # {section: {eid_str: [{"claim": "", "entailment_prediction": 0 or 1}, ...]} }
    
    if os.path.exists(savefile):
        print('Save file exist!')
        claims_score = json.load(open(savefile))
    else:
        claims_score = {}
        for section in SECTION_DIVISIONS:
            claims_score[section] = {}
            for x in output_data:
                eid_str = str(x['example_id'])
                claims_score[section][eid_str] = []
                
    # # Load model and tokenizer  
    if eval_model == 'TRUE':
        print('Loading models...')
        device = "cuda"
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory())
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        autoais_model.to(device)
    
    elif eval_model == 'Mistral':
        print('Loading models...')
        model = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_model_len=16384, gpu_memory_utilization=0.8, download_dir=cache_dir)
        tokenizer = model.get_tokenizer()
        
        prompt_template = json.load(open(prompt_file, 'r'))


    wrong_format_count = 0
    new_generation_count = 0
    for section in SECTION_DIVISIONS:
        if mode == 'claim_recall':
            if dataset_name == 'meqsum':
                text_key = 'output'
                subclaim_key = 'reference'
            else:
                if args.use_persection_claims:
                    text_key = f'output_{section}'
                    subclaim_key = f'subclaims_reference_{section}'
                else:
                    text_key = 'output'
                    subclaim_key = 'subclaims_reference'
                
        elif mode == 'claim_precision':
            if dataset_name == 'meqsum':
                text_key = 'reference'
                subclaim_key = 'output'
            else:
                if args.use_persection_claims:
                    text_key = f'reference_{section}'
                    subclaim_key = f'subclaims_output_{section}'
                else:
                    text_key = 'reference'
                    subclaim_key = 'subclaims_output'
                
        elif mode == 'same':
            assert dataset_name == 'meqsum'
            text_key = 'output'
            subclaim_key = 'reference'
            
        
        for item in output_data:
            eid_str = str(item['example_id'])
            text = remove_citations(item[text_key])
            
            if dataset_name == 'meqsum':
                claims = [item[subclaim_key]]
            else:
                claims = item[subclaim_key]
                
                if len(claims) == 0:
                    # skip empty claims
                    claims_score[section][eid_str] = []
                    continue
                
                if len(text) == 0:
                    # score is 0 for all claims
                    claims_score[section][eid_str] = [{"claim": claim, "entailment_prediction":0 } for claim in claims]
                    continue 
                
            existing_count = sum([1 for x in claims_score[section][eid_str] if "entailment_prediction" in x])
            if existing_count == len(claims):
                continue
            
            print(f"Prediction not complete for {section}-{eid_str}")
            
            if len(claims_score[section][eid_str]) < len(claims):
                claims_score[section][eid_str] = [{} for _ in claims]
                
            # TRUE
            if eval_model == 'TRUE':
                for cid, claim in enumerate(claims):
                    if "entailment_prediction" in claims_score[section][eid_str][cid]:
                        continue
                    
                    entail_score = _run_nli_autoais(text, claim, autoais_model, autoais_tokenizer)
                    claims_score[section][eid_str][cid] = {
                        'claim': claim,
                        'entailment_prediction': entail_score
                    }
                    print(f"{entail_score} {claim}")
            
            
            # Mistral
            elif eval_model == 'Mistral':
                new_gen_flag = False
                for cid, claim in enumerate(claims):
                    if "entailment_prediction" in claims_score[section][eid_str][cid]:
                        continue
                    
                    new_gen_flag = True
                    
                    prompt = deepcopy(prompt_template)
                    prompt[-1]['content'] = prompt[-1]['content'].replace('__TEXT__', text).replace('__CLAIM__', claim)
                    prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False)

                    count = 15
                    while count > 0:
                        decoded = call_llm(model, prompt_str, max_new_tokens=max_new_tokens)
                        response_text = decoded[0].split('[/INST] ')[-1]
                        
                        explanation, prediction = "", -1
                        for line in response_text.split('\n'):
                            if "prediction: 1" in line and prediction == -1:
                                prediction = 1
                            
                            elif "prediction: 0" in line and prediction == -1:
                                prediction = 0
                                
                            if "explanation: " in line and explanation == "":
                                explanation = line.split('explanation: ')[-1]
                                
                        if prediction != -1:
                            break 
                        else:
                            print('CANNOT READ PREDICTION: ', response_text)
                            count -= 1
                        
                    if prediction != -1:
                        claims_score[section][eid_str][cid] = {
                            'claim': claim,
                            'explanation': explanation,
                            'entailment_prediction': prediction,
                        }
                        print(f"{prediction} {claim}")
                    else:
                        claims_score[section][eid_str][cid] = {
                            'claim': claim,
                            'explanation': explanation,
                        }
                        wrong_format_count += 1
            
                new_generation_count += int(new_gen_flag)
                if new_gen_flag and new_generation_count % 3 == 0:
                    # save every 20 steps
                    print('Saving results..')
                    json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)
        
    json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)
    
    print(f"WRONG FORMAT COUNT: {wrong_format_count}")
    