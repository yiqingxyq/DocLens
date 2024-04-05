import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re


def get_claim_scores(claim_scores):
    scores = {}
    total_entail_count, total_claim_count = 0, 0
    for section in claim_scores:
        entail_count, claim_count = 0, 0
        for eid_str in claim_scores[section]:
            entail_count += len([1 for x in claim_scores[section][eid_str] if x['entailment_prediction'] == 1])
            claim_count += len(claim_scores[section][eid_str])
            
        if claim_count > 0:
            scores[section] = entail_count / claim_count
        else:
            scores[section] = 0
            
        total_entail_count += entail_count
        total_claim_count += claim_count 
    
    if total_claim_count > 0:
        scores['total'] = total_entail_count / total_claim_count 
    else:
        scores['total'] = 0
    
    return scores

def get_citation_scores(citation_scores):
    scores = {'citation_precision': {}, 'citation_recall': {}}
    total_correct_citation_count, total_citation_count = 0, 0
    total_correct_sent_count, total_sent_count = 0, 0
    for section in citation_scores:
        correct_citation_count, citation_count = 0, 0
        correct_sent_count, sent_count = 0, 0
        for eid_str in citation_scores[section]:
            for sent_pred_dict in citation_scores[section][eid_str]:
                sent_count += 1
                citation_count += len(sent_pred_dict['citations'])
                
                if sent_pred_dict['entailment_prediction'] == 1:
                    correct_sent_count += 1
                    correct_citation_count += len([x for x in sent_pred_dict['citations'] if x in sent_pred_dict['provenance']])
                    
        if citation_count > 0:
            scores['citation_precision'][section] = correct_citation_count / citation_count
        else:
            scores['citation_precision'][section] = 0
            
        if sent_count > 0:
            scores['citation_recall'][section] = correct_sent_count / sent_count
        else:
            scores['citation_recall'][section] = 0
        
        total_sent_count += sent_count
        total_correct_sent_count += correct_sent_count
        total_citation_count += citation_count
        total_correct_citation_count += correct_citation_count

    if total_citation_count > 0:
        scores['citation_precision']['total'] = total_correct_citation_count / total_citation_count
    else:
        scores['citation_precision']['total'] = 0
        
    if total_sent_count > 0:
        scores['citation_recall']['total'] = total_correct_sent_count / total_sent_count
    else:
        scores['citation_recall']['total'] = 0
        
    return scores
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    
    # evaluation setting
    parser.add_argument("--eval_claim_recall", action="store_true", default=False, help="Compute claim recall")
    parser.add_argument("--eval_claim_precision", action="store_true", default=False, help="Compute claim precision")
    parser.add_argument("--eval_citations", action="store_true", default=False, help="Compute citation recall & precision")
    
    # evaluation model
    parser.add_argument("--eval_model", type=str, choices=['TRUE', 'Mistral', 'GPT'])
    
    args = parser.parse_args()
    
    result_file, dataset_name, eval_model = args.result_file, args.dataset_name, args.eval_model
    
    if eval_model == 'GPT':
        if dataset_name == 'meqsum':
            score_file_claim_recall = result_file.replace('.json', '.claim_scores')
            score_file_claim_precision = result_file.replace('.json', '.output_claim_scores')
            if eval_model != 'TRUE':
                score_file_claim_same = result_file.replace('.json', '.same_scores') 
        else:
            score_file_claim_recall = result_file.replace('.json', '.claim_min1max30.claim_scores')
            score_file_claim_precision = result_file.replace('.json', '.output_claim_min1max30.output_claim_scores')
            
        score_file_citations = result_file.replace('.json', '.citations.score')
        
    else:
        if dataset_name == 'meqsum':
            score_file_claim_recall = result_file.replace('.json', f'.claim_scores.{eval_model}')
            score_file_claim_precision = result_file.replace('.json', f'.output_claim_scores.{eval_model}')
            if eval_model != 'TRUE':
                score_file_claim_same = result_file.replace('.json', f'.same_scores.{eval_model}')
        else:
            score_file_claim_recall = result_file.replace('.json', f'.claim_min1max30.claim_scores.{eval_model}')
            score_file_claim_precision = result_file.replace('.json', f'.output_claim_min1max30.output_claim_scores.{eval_model}')
            
        score_file_citations = result_file.replace('.json', f'.citations.score.{eval_model}')

            
    SCORES = {k:{'total': 0} for k in ['claim_recall', 'claim_precision', 'citation_recall', 'citation_precision']}
            
    if args.eval_claim_recall:
        claim_recall_scores = json.load(open(score_file_claim_recall, 'r'))
        if dataset_name == 'meqsum' and eval_model != 'TRUE':
            claim_same_scores = json.load(open(score_file_claim_same, 'r'))
            for section in claim_recall_scores:
                for eid_str in claim_recall_scores[section]:
                    for x_recall, x_same in zip(claim_recall_scores[section][eid_str], claim_same_scores[section][eid_str]):
                        x_recall['entailment_prediction'] = int(x_recall['entailment_prediction'] == 1 or x_same['entailment_prediction'] == 1)
            
        SCORES['claim_recall'] = get_claim_scores(claim_recall_scores)
        
    if args.eval_claim_precision:
        claim_precision_scores = json.load(open(score_file_claim_precision, 'r'))
        if dataset_name == 'meqsum' and eval_model != 'TRUE':
            claim_same_scores = json.load(open(score_file_claim_same, 'r'))
            for section in claim_precision_scores:
                for eid_str in claim_precision_scores[section]:
                    for x_precision, x_same in zip(claim_precision_scores[section][eid_str], claim_same_scores[section][eid_str]):
                        x_precision['entailment_prediction'] = int(x_precision['entailment_prediction'] == 1 or x_same['entailment_prediction'] == 1)
            
        SCORES['claim_precision'] = get_claim_scores(claim_precision_scores)
        
    if args.eval_citations:
        citation_scores = json.load(open(score_file_citations, 'r'))
        
        SCORES.update(get_citation_scores(citation_scores))
        
    print(json.dumps(SCORES, indent=4))