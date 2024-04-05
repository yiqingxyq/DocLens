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

MIN_CLAIM = 1
MAX_CLAIM = 30

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

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


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--eval_file', default=None, help='filename of the eval_data .json.')
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    
    # claim generation setting
    parser.add_argument('--mode', type=str, default='reference_claims', choices=['reference_claims', 'output_claims'],
                        help='whether to generate claims for the references or outputs')
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    
    # claim generation model
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    
    args = parser.parse_args()
    
    eval_file, result_file, mode, prompt_file, max_new_tokens = args.eval_file, args.result_file, args.mode, args.prompt_file, args.max_new_tokens
    
    if args.azure:
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        CLAIM_EXTRACTOR_NAME = CLAIM_EXTRACTOR_DEPLOY_NAME = "gpt-4-1106-preview"
        # CLAIM_EXTRACTOR_NAME = CLAIM_EXTRACTOR_DEPLOY_NAME = "gpt-35-turbo"
    else:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = os.environ.get("OPENAI_API_KEY_OPENAI")
        CLAIM_EXTRACTOR_NAME = "gpt-4-1106-preview"
    
    if mode == 'reference_claims':
        assert eval_file is not None
    
    existing_set = set()
    if mode == 'reference_claims':
        claim_file = eval_file.replace('.json', f'.claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        input_data_file = claim_file if os.path.exists(claim_file) else eval_file
        
        if args.use_persection_claims:
            text_keys = [f"reference_{section}" for section in SECTION_DIVISIONS]
            claim_keys = [f"subclaims_reference_{section}" for section in SECTION_DIVISIONS]
            prompt_template_dict = {f"reference_{section}": json.load(open(prompt_file.replace('persection', section))) for section in SECTION_DIVISIONS}
        else:
            text_keys = ['reference']
            claim_keys = ['subclaims_reference']
            prompt_template_dict = {'reference': json.load(open(prompt_file))}
    
    else:
        claim_file = result_file.replace('.json', f'.output_claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        input_data_file = claim_file if os.path.exists(claim_file) else result_file
        
        if args.use_persection_claims:
            text_keys = [f"output_{section}" for section in SECTION_DIVISIONS]
            claim_keys = [f"subclaims_output_{section}" for section in SECTION_DIVISIONS]
            prompt_template_dict = {f"output_{section}": json.load(open(prompt_file.replace('persection', section))) for section in SECTION_DIVISIONS}
        else:
            text_keys = ['output']
            claim_keys = ['subclaims_output']
            prompt_template_dict = {'output': json.load(open(prompt_file))}
            
    data = json.load(open(input_data_file))
            
    for k in prompt_template_dict:
        prompt_template_dict[k][0]['content'] = prompt_template_dict[k][0]['content'].replace('MIN_CLAIM', str(MIN_CLAIM)).replace('MAX_CLAIM', str(MAX_CLAIM))
        
    if mode == 'reference_claims':
        # copy results from data to result_file 
        result_data = json.load(open(result_file))
        eid2result_item = {x['example_id']:x for x in result_data}
            
    wrong_format_count = 0
    total_count = 0
    for item in data:
        for text_key, claim_key in zip(text_keys, claim_keys):
            if claim_key in item and type(item[claim_key]) == list:
                continue 
            
            if mode == 'reference_claims':
                if item['example_id'] not in eid2result_item:
                    continue
            
            text = item[text_key]
            prompt = deepcopy(prompt_template_dict[text_key])
            prompt[-1]['content'] = text 
            
            if len(text) == 0:
                item[claim_key] = []
                continue
            
            if args.azure:
                response = completion_with_backoff(
                    engine=CLAIM_EXTRACTOR_DEPLOY_NAME, model=CLAIM_EXTRACTOR_NAME, messages=prompt, max_tokens=max_new_tokens
                )
            else:
                response = completion_with_backoff(
                    model=CLAIM_EXTRACTOR_NAME, messages=prompt, max_tokens=max_new_tokens
                )
                
            try:
                claims_text = response['choices'][0]['message']['content']
                subclaims_list = re.split('Claim [0-9]+: ', claims_text.replace('\n',''))[1:]
                        
                print(item['example_id'], text_key)
                print('='*50)
                for claim in subclaims_list:
                    print(claim)
                print('='*50)
                item[claim_key] = subclaims_list
            
            except:
                print(f"Wrong format for {item['example_id']}-{text_key}")
                wrong_format_count += 1
                
            total_count += 1
        
            if total_count % 5 == 0:
                print(f'Saving to files: {claim_file}..')
                json.dump(data, open(claim_file, 'w'), indent=4)
                
    if total_count > 0:
        print(f'Saving to files: {claim_file}..')
        json.dump(data, open(claim_file, 'w'), indent=4)


    if mode == 'reference_claims':
        # copy results from data to result_file 
        result_data = json.load(open(result_file))
        eid2item = {x['example_id']:x for x in data}
        for result_item in result_data:
            item = eid2item[result_item['example_id']]
            for claim_key in claim_keys:
                if claim_key in item:
                    result_item[claim_key] = item[claim_key]
                    
        output_file = result_file.replace('.json', f'.claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        print(f'Saving to files: {output_file}..')
        json.dump(result_data, open(output_file, 'w'), indent=4)