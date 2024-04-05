import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re

from nltk import sent_tokenize

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
    parser.add_argument("--split_method", type=str, choices=['sent', 'citation'], help="Split the generation output by sent/citation idx")
    parser.add_argument("--max_citation_num", type=int, default=10)
    parser.add_argument("--get_persection_score", action="store_true", default=False, help="Compute the scores for each section")
    
    # evaluation model
    parser.add_argument('--prompt_file', default=None, help='filename of the prompt dict .json.')
    parser.add_argument("--eval_model", type=str, default="Mistral", choices=['TRUE', 'Mistral'])
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max number of new tokens to generate in one step")
    parser.add_argument("--cache_dir", type=str, default="llama_models")
    
    args = parser.parse_args()
    
    result_file, dataset_name, split_method, max_citation_num = args.result_file, args.dataset_name, args.split_method, args.max_citation_num
    prompt_file, eval_model, max_new_tokens, cache_dir = args.prompt_file, args.eval_model, args.max_new_tokens, args.cache_dir
    
    savefile = result_file.replace('.json', f'.citations.score.{eval_model}')
    
    if not args.get_persection_score:
        SECTION_DIVISIONS = ['full']
        
    output_data = json.load(open(result_file, 'r')) # a list of dicts
    
    print( f"Saving scores to {savefile.split('/')[-1]}..") # {section: {eid_str: [{"send_id": "", "output": "", ... "entailment_prediction": 0 or 1}, ...]} }
    
    if os.path.exists(savefile):
        print('Save file exist!')
        citations_score = json.load(open(savefile, 'r'))
    else:
        citations_score = {}
        for section in SECTION_DIVISIONS:
            citations_score[section] = {}
            for x in output_data:
                eid_str = str(x['example_id'])
                citations_score[section][eid_str] = []
                
    prompt_template = json.load(open(prompt_file, 'r'))
    
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

    
    # run entailment
    wrong_format_count = 0
    wrong_entailment_count = 0
    sent_count = 0
    new_generation_count = 0
    for section in SECTION_DIVISIONS:
        
        if args.get_persection_score:
            output_key = f'output_{section}'
        else:
            output_key = 'output'
        
        for item in output_data:
            eid_str, input_text, output_text = str(item['example_id']), item['input'], item[output_key]
                
            if output_text == "":
                # skip empty note
                citations_score[section][eid_str] = []
                continue
            
            # preprocess input (split output note into sents, split input_text by idx)
            if dataset_name == 'meqsum':
                # only one sent in the generation output
                sents = [output_text]
            elif split_method == 'sent':
                sents = sent_tokenize(output_text)
            elif split_method == 'citation':
                clean_sents = re.split("[\[\d+\]]+", output_text)[:-1] # remove the last split without citations
                citations = re.findall("[\[\d+\]]+", output_text)
                sents = [s+c for s,c in zip(clean_sents, citations)]
                if len(sents) == 1:
                    print('Citation not found')
                    wrong_format_count += 1
                    sent_count += 1
                    citations_score[section][eid_str] = [{
                        "sent_id": 0,
                        "output": "",
                        "citations": [],
                        "cited_sents": [],
                        "entailment_prediction": 0,
                        "explanation": "",
                        "provenance": "",
                    }]
                    
                    continue
                    
            sents = [" ".join(s.split()) for s in sents] # output sents w/ citations
            target_sents = [remove_citations(sent) for sent in sents]
            
            # split input text by citations
            input_sents = re.split("\[\d+\]", input_text)[1:] # the sent is after its citation idx
            citations = re.findall("\[\d+\]", input_text)
            input_sents = [" ".join(s.split()) for s in input_sents]
            docs = {int(citation[1:-1]): sent for sent, citation in zip(input_sents, citations)}
            
            # run entailment
            sent_count += len(sents)
            new_gen_flag = False
            if len(citations_score[section][eid_str]) < len(sents):
                citations_score[section][eid_str] = [{} for _ in sents]
                
            for sent_id, sent in enumerate(sents):
                if 'entailment_prediction' in citations_score[section][eid_str][sent_id]: 
                    continue
                
                new_gen_flag = True
                
                target_sent = target_sents[sent_id] # The output sent
                
                if dataset_name == 'mimic' and eval_model == 'TRUE':
                    # improve the performance of TRUE: remove "1.", "2." ... in mimic generation
                    pattern = r'\d+. '
                    target_sent = re.sub(pattern, '', target_sent)

                # Find references
                ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)] # In our setting the citation starts from 0
                ref = list(set(ref)) # there could be repeated ref
                print('-'*20, f'eid_str: {eid_str}, Sentence idx: {sent_id}', '-'*20)
                print(f"For `{sent}`, find citations {ref}")
                
                if len(ref) == 0:
                    # No citations
                    # Reach the next citation
                    for next_sent_id in range(sent_id+1, len(sents)):
                        next_sent = sents[next_sent_id]
                        next_target_sent = target_sents[next_sent_id]
                        ref = [int(r[1:]) for r in re.findall(r"\[\d+", next_sent)]
                        if len(ref) > 0:
                            break
                    print(f"For `{sent}`, find citations {ref}")
                
                if len(ref) == 0 or any([ref_id >= len(docs) for ref_id in ref]):
                    # No citations or Citations out of range
                    print(f"Invalid citation format: {ref}")
                    wrong_format_count += 1
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": [],
                        "entailment_prediction": 0,
                    }
                    
                    continue
                    
                ref = ref[:args.max_citation_num]
                
                joint_passage = ' '.join([docs[psgs_id] for psgs_id in ref])
                joint_passage_w_idx = '\n'.join([ f"[{present_id}] " + docs[psgs_id] for present_id, psgs_id in enumerate(ref)])
                    
                if eval_model == 'Mistral':
                    
                    prompt = deepcopy(prompt_template)
                    prompt[-1]['content'] = prompt[-1]['content'].replace('__TEXT__', joint_passage_w_idx).replace('__CLAIM__', target_sent)
                    prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False)
                    
                    count = 15
                    while count > 0:
                        decoded = call_llm(model, prompt_str, max_new_tokens=max_new_tokens)
                        response_text = decoded[0].split('[/INST] ')[-1]
                        
                        explanation, prediction, provenance = "", -1, []
                        if len(ref) == 1:
                            provenance = ref # if there is only one citation, do not need to print provenance
                        
                        for line in response_text.split('\n'):
                            if "prediction: 1" in line and prediction == -1:
                                prediction = 1
                            
                            elif "prediction: 0" in line and prediction == -1:
                                prediction = 0
                                
                            if "explanation: " in line and explanation == "":
                                explanation = line.split('explanation: ')[-1]
                                
                            if "provenance: " in line and len(provenance) == 0:
                                provenance_str = line.split('explanation: ')[-1]
                                provenance = re.findall('\d+', provenance_str)
                                provenance = [int(x) for x in provenance]
                                print(f"Find provenance: {provenance}")
                                provenance = [ref[x] for x in provenance if x in range(len(ref))] # return the orig ref_id
                                print(f"Find valid provenance: {provenance}")
                                
                        if prediction == -1:
                            print('CANNOT READ PREDICTION: ', response_text)
                            count -= 1
                        elif prediction == 1 and len(provenance)==0:
                            print('WRONG PROVENANCE: ', response_text)
                            count -= 1
                        else:
                            break
                            
                    if prediction == -1 or (prediction == 1 and len(provenance)==0):
                        wrong_entailment_count += 1
                        citations_score[section][eid_str][sent_id] = {
                            "sent_id": sent_id,
                            "output": sent,
                            "citations": ref,
                            "cited_sents": joint_passage_w_idx,
                            "response": response_text
                        }
                    
                    else:
                        citations_score[section][eid_str][sent_id] = {
                            "sent_id": sent_id,
                            "output": sent,
                            "citations": ref,
                            "cited_sents": joint_passage_w_idx,
                            "entailment_prediction": prediction,
                            "explanation": explanation,
                            "provenance": provenance,
                        }
                        print(f"cited_sents: {joint_passage_w_idx}")
                        print(f"entailment_prediction: {prediction}; provenance: {provenance}")
                            
                elif eval_model == 'TRUE':
                    # check claim recall 
                    joint_entail = _run_nli_autoais(joint_passage, target_sent, autoais_model, autoais_tokenizer)
                    
                    # claim recall = 0
                    if joint_entail == 0:
                        citations_score[section][eid_str][sent_id] = {
                            "sent_id": sent_id,
                            "output": sent,
                            "citations": ref,
                            "cited_sents": joint_passage,
                            "entailment_prediction": 0
                        }
                        print(f"cited_sents: {joint_passage}")
                        print("entailment_prediction: 0")
                    else:
                        # claim recall = 1
                        citations_score[section][eid_str][sent_id] = {
                            "sent_id": sent_id,
                            "output": sent,
                            "citations": ref,
                            "cited_sents": joint_passage,
                            "entailment_prediction": 1
                        }
                        
                        if len(ref) == 1:
                            citations_score[section][eid_str][sent_id]['provenance'] = ref
                        else:
                            # check claim precision 
                            provenance = []
                            for psgs_id in ref:
                                # condition A
                                passage = docs[psgs_id]
                                nli_result = _run_nli_autoais(passage, target_sent, autoais_model, autoais_tokenizer)
                                
                                # condition B
                                if not nli_result:
                                    subset_exclude = deepcopy(ref)
                                    subset_exclude.remove(psgs_id)
                                    passage = '\n'.join([docs[pid] for pid in subset_exclude])
                                    nli_result = _run_nli_autoais(passage, target_sent, autoais_model, autoais_tokenizer)
                                    if not nli_result: # psgs_id is necessary
                                        provenance.append(psgs_id)
                                else:
                                    provenance.append(psgs_id)
                                    
                            citations_score[section][eid_str][sent_id]['provenance'] = provenance
                        
                        print(f"cited_sents: {joint_passage}")
                        print(f"entailment_prediction: 1; provenance: {citations_score[section][eid_str][sent_id]['provenance']}")
                
            new_generation_count += int(new_gen_flag)
            if new_gen_flag and new_generation_count % 3 == 0:
                print('Saving results..')
                json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True) 
                
    # save results
    json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True)
    
    print(f"Wrong format count: {wrong_format_count}/{sent_count}")
    print(f"Wrong entailment count: {wrong_entailment_count}/{sent_count}")
    