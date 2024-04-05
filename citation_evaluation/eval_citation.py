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

from nltk import sent_tokenize


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
    parser.add_argument("--split_method", type=str, choices=['sent', 'citation'], help="Split the generation output by sent/citation idx")
    parser.add_argument("--max_citation_num", type=int, default=10)
    parser.add_argument("--get_persection_score", action="store_true", default=False, help="Compute the scores for each section")
    
    # evaluation model
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")
    
    args = parser.parse_args()
    
    result_file, dataset_name, split_method, max_citation_num, prompt_file, max_new_tokens = args.result_file, args.dataset_name, args.split_method, args.max_citation_num, args.prompt_file, args.max_new_tokens
    savefile = result_file.replace('.json', '.citations.score')
    
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
    
    TEXT_NAME = {
        'acibench': {'output_sent_name': 'sentence_in_note', 'cited_input_name': 'conversational_turns'},
        'mimic': {'output_sent_name': 'sentence_in_summary', 'cited_input_name': 'sentences_in_the_radiology_report'},
        'meqsum': {'output_sent_name': 'short_question', 'cited_input_name': 'sentences_in_the_long_question'},
    }
    
    # run entailment
    wrong_format_count = 0
    wrong_entailment_count = 0
    sent_count = 0
    new_generation_count = 0
    for section in SECTION_DIVISIONS:
        if args.get_persection_score:
            output_key = f'output_{section}'
            prompt_template = json.load(open(prompt_file.replace('persection', section), 'r'))
        else:
            output_key = 'output'
            prompt_template = json.load(open(prompt_file, 'r'))
            
        for i in range(1,len(prompt_template)-1):
            prompt_template[i]['content'] = json.dumps(prompt_template[i]['content'])
            
        if dataset_name in TEXT_NAME:
            output_sent_name, cited_input_name = TEXT_NAME[dataset_name]["output_sent_name"], TEXT_NAME[dataset_name]["cited_input_name"]
        else:
            output_sent_name, cited_input_name = "generated_sentence", "sentence_in_clinical_report"
        
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
                        "provenance": [],
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
                if "entailment_prediction" in citations_score[section][eid_str][sent_id]:
                    continue
                
                new_gen_flag = True
                
                target_sent = target_sents[sent_id] # The output sent

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
                        "explanation": "",
                        "provenance": [],
                    }
                    
                    continue
                    
                ref = ref[:args.max_citation_num]

                # compute citation scores 
                if dataset_name == 'acibench':
                    joint_passage = []
                    for psgs_id in ref:
                        speaker = re.findall(r"\[[a-z,\s,_]+\]", docs[psgs_id])[0][1:-1]
                        content = re.sub(r"\[[a-z,\s,_]+\] ", "", docs[psgs_id])
                        joint_passage.append({
                            "idx": str(psgs_id),
                            "speaker": speaker,
                            "content": content
                        })
                else:
                    joint_passage = []
                    for psgs_id in ref:
                        joint_passage.append({
                            "idx": str(psgs_id),
                            "content": docs[psgs_id]
                        })
                    
                print(joint_passage)
                
                prompt = deepcopy(prompt_template)
                prompt[-1]['content'] = json.dumps({
                    output_sent_name: target_sent,
                    cited_input_name: joint_passage
                })
                
                if args.azure:
                    response = completion_with_backoff(
                        engine=EVALUATOR_DEPLOY_NAME, model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens
                    )
                else:
                    response = completion_with_backoff(
                        model=EVALUATOR_NAME, messages=prompt, max_tokens=max_new_tokens
                    )
                
                if len(response) == 0:
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "response": "",
                    }
                    print('No response from the evaluator model')
                    wrong_entailment_count += 1
                    
                    continue
                else:
                    response_content = response['choices'][0]['message']['content']
                
                try:
                    response_dict = json.loads(response_content) # entailment_prediction, explanation, provenance
                    print(json.dumps(response_dict, indent=4))
                    
                    response_dict.update({
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "entailment_prediction": response_dict['entailment_prediction'],
                        "explanation": response_dict['explanation'],
                        "provenance": response_dict['provenance'],
                    })
                    
                    citations_score[section][eid_str][sent_id] = response_dict
                except:
                    wrong_entailment_count += 1
                    print('!'*10, 'Cannot convert to json format', '!'*10)
                    print(response_content)
                    citations_score[section][eid_str][sent_id] = {
                        "sent_id": sent_id,
                        "output": sent,
                        "citations": ref,
                        "cited_sents": joint_passage,
                        "response": response_content,
                    }
            
            new_generation_count += int(new_gen_flag)
            if new_gen_flag and new_generation_count % 3 == 0:
                print('Saving results..')
                json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True)

    # save results
    json.dump(citations_score, open(savefile, 'w'), indent=4, sort_keys=True)
    
    print(f"Wrong format count: {wrong_format_count}/{sent_count}")
    print(f"Wrong entailment count: {wrong_entailment_count}/{sent_count}")