import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re

import openai
import openai.error


SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def completion_with_backoff(**kwargs):
    is_ok = False
    retry_count = 0
    while not is_ok:
        retry_count += 1
        try:
            response = openai.ChatCompletion.create(**kwargs)
            is_ok = True
        except openai.error.RateLimitError as error:
            if retry_count <= 30:
                if retry_count % 10 == 0:
                    print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(10)
                continue
            else:
                return {}
        except openai.error.InvalidRequestError as error:
            if 'maximum context length' in error._message:
                if retry_count <= 3:
                    print(f"reduce max_tokens by 500")
                    kwargs['max_tokens'] = kwargs['max_tokens'] - 500
                    continue
                else:
                    print(error)
                    return {}
            else:
                print(error)
                return {}
        except Exception as error:
            print(error)
            return {}
    return response


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    
    # evaluation setting
    parser.add_argument("--mode", type=str, default="claim_recall", choices=['claim_recall','claim_precision','same'])
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    
    # evaluation model
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    
    args = parser.parse_args()
    
    result_file, dataset_name, mode, prompt_file, max_new_tokens = args.result_file, args.dataset_name, args.mode, args.prompt_file, args.max_new_tokens

    # API setup 
    if args.azure:
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        EVALUATOR_NAME = EVALUATOR_DEPLOY_NAME = "gpt-4-1106-preview" 
        # EVALUATOR_NAME = EVALUATOR_DEPLOY_NAME = "gpt-35-turbo"
    else:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        EVALUATOR_NAME = "gpt-4-1106-preview"
    
    if mode == 'claim_recall':
        savefile = result_file.replace('.json', '.claim_scores')
    elif mode == 'claim_precision':
        savefile = result_file.replace('.json', '.output_claim_scores')
    elif mode == 'same':
        # For MeQSum, since the output only contains one line, we check:
        # (1) whether the output question is a specification of the reference question
        # (2) whether the reference question is a specification of the output question
        # (3) whether the two questions are essentially the same
        assert dataset_name == 'meqsum'
        savefile = result_file.replace('.json', '.same_scores')
        
    if not args.use_persection_claims:
        SECTION_DIVISIONS = ['full']
    
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

    prompt_template = json.load(open(prompt_file, 'r'))
    for i in range(1,len(prompt_template)-1):
        prompt_template[i]['content'] = json.dumps(prompt_template[i]['content'])

    TEXT_NAME = {
        'acibench': 'clinical_note',
        'mimic': 'radiology_report',
    }

    wrong_format_count = 0
    new_generation_count = 0
    for section in SECTION_DIVISIONS:
        if args.mode == 'claim_recall':
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
                
        elif args.mode == 'claim_precision':
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
                
        elif args.mode == 'same':
            text_key = 'output'
            subclaim_key = 'reference'
            
        text_name = TEXT_NAME[dataset_name] if dataset_name in TEXT_NAME else "clinical_report"
        
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
            
            # claim style = jsonexp_2shot
            if len(claims_score[section][eid_str]) == len(claims):
                continue
                
            print('Do not exist:', section, eid_str)
            prompt = deepcopy(prompt_template)
            
            if dataset_name == 'meqsum':
                prompt[-1]['content'] = json.dumps({
                    "question A": text,
                    "question B": item[subclaim_key]
                })
            else:
                prompt[-1]['content'] = json.dumps({
                    text_name: text,
                    "claims": claims
                })
                    
            if args.azure:
                response = completion_with_backoff(
                    engine=EVALUATOR_DEPLOY_NAME, model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens
                )
            else:
                response = completion_with_backoff(
                    model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens
                )
            
            new_generation_count += 1
            
            try:
                judgment_dict = json.loads(response['choices'][0]['message']['content'])
                if dataset_name == 'meqsum':
                    judgment_dict['entailment_prediction'] = judgment_dict['prediction']
                    claims_score[section][eid_str] = [judgment_dict]
                    print(f"question A: {text}")
                    print(f"question B: {claims[0]}")
                    print(f"{judgment_dict['entailment_prediction']} {judgment_dict['explanation']}")
                else:
                    claims_score[section][eid_str] = judgment_dict
                    for cid, d in enumerate(claims_score[section][eid_str]):
                        print(f"{d['entailment_prediction']} Claim {cid}: {d['claim']}")
            except:
                print('CANNOT CONVERT TO JSON')
                print(response)
                wrong_format_count += 1
            
            if new_generation_count % 5 == 0:
                # save every 5 steps
                print('Saving results..')
                json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)
        
    json.dump(claims_score, open(savefile, 'w'), indent=4, sort_keys=True)
    
    print(f"WRONG FORMAT COUNT: {wrong_format_count}")
    