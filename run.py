import argparse
import os
import yaml
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from transformers import GPT2TokenizerFast

import openai
import openai.error

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


def get_prompt_len(prompt, tokenizer):
    prompt_str = ""
    for p in prompt:
        for k in p:
            prompt_str = prompt_str + f"{k} {p[k]} "
        prompt_str = prompt_str + ' \n '
            
    return len(tokenizer.tokenize( prompt_str )) + 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--config_file", type=str, default=None)
    
    # Save setting
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--result_dir", type=str, help="Path to the result file", default='result')
    parser.add_argument("--dataset_and_tag", type=str, default=None, help="Name of the dataset and tag (for saving)")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    
    # Text generation model
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--deploy_name", type=str, default=None, help="Deployname of the model to use")
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    # Decoding
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file)) if args.config_file is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    
    if "32k" in args.model:
        print("Change the max length to 32768 for GPT-4-32k.")
        max_length = 32768
    elif "16k" in args.model:
        print("Change the max length to 16384 for ChatGPT-16k.")
        max_length = 16384
    elif "gpt-4" in args.model and "preview" in args.model:
        print("Change the max length to 128k for GPT-4-turbo.")
        max_length = 128000
    elif "gpt-4" in args.model:
        print("Change the max length to 8192 for GPT-4.")
        max_length = 8192
    elif "35" in args.model or "3.5" in args.model:
        print("Change the max length to 4096 for ChatGPT.")
        max_length = 4096
    else:
        max_length = 2048
                
    # API setup 
    if args.azure:
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_type = "azure"
        openai.api_version = "2023-05-15" 
    else:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
    # compute prompt length
    tokenizer = GPT2TokenizerFast.from_pretrained('RaymondLi/gpt-4-tokenizer') 
        
    # Random seed (will be used for quick_test)
    np.random.seed(args.seed)
        
    # Load data
    prompt_template = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file)) # a list of dicts. The keys include: input, example_id, reference
    
    # use the first k shot
    if 'dummy' not in args.prompt_file:
        prompt_template = [prompt_template[0]] + prompt_template[1:args.shot*2+1] + [prompt_template[-1]]
        
    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]
        
    # Generating save name
    if 'dummy' in args.prompt_file:
        shot_name = 0
    else:
        shot_name = args.shot
        
    model_name = args.model.replace("/",'_')
    savename = f"{args.dataset_and_tag}-{model_name}-shot{shot_name}"
    if args.azure:
        savename += "-azure"
    if args.quick_test is not None:
        savename += f"-quick_test{args.quick_test}"
        
    os.makedirs(args.result_dir, exist_ok=True)
    savefile = os.path.join(args.result_dir, savename + ".json")
        
    if args.quick_test is None:
        # load existing results, skip the items with results 
        if os.path.exists(savefile):
            print(f'Savefile exists: {savefile}')
            save_data = json.load(open(savefile, 'r'))
            assert len(save_data) == len(eval_data)
            eval_data = save_data
        
    new_generation_count = 0
    for idx, item in enumerate(tqdm(eval_data)):
        if 'output' in item:
            if type(item['output']) == str:
                # skip items with results
                continue 

        prompt = deepcopy(prompt_template)
        prompt[-1]['content'] = item['input']
        prompt_len = get_prompt_len(prompt, tokenizer)
        
        if "gpt-4" in args.model and "preview" in args.model:
            max_new_tokens = min(args.max_new_tokens, 4096)
            print( f"max_new_tokens = min({args.max_new_tokens}, 4096)")
        else:
            max_new_tokens = min(args.max_new_tokens, max_length - prompt_len - 10)
            print( f"max_new_tokens = min({args.max_new_tokens}, {max_length-prompt_len-10}={max_length}-{prompt_len}-10)")
        
        if args.azure:
            deploy_name = args.deploy_name if args.deploy_name else args.model
            response = completion_with_backoff(
                engine=deploy_name, model=args.model, messages=prompt, max_tokens=max_new_tokens
            )
        else:
            response = completion_with_backoff(
                model=args.model, messages=prompt, max_tokens=max_new_tokens
            )
            
        try:
            output = response['choices'][0]['message']['content']
            item['output'] = output
            new_generation_count += 1
        except:
            print(f"Generation Error for sample: {idx}")
            
        if new_generation_count % 10 == 0:
            json.dump(eval_data, open(savefile, "w"), indent=4)
            
    
    json.dump(eval_data, open(savefile, "w"), indent=4)
    
if __name__ == "__main__":
    main()
