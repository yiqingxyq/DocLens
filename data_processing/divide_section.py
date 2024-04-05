import argparse
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from sectiontagger import SectionTagger
section_tagger = SectionTagger()

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, help="Path to the result file")
    args = parser.parse_args()
    
    if 'persection' in args.result_file:
        data_list = [json.load(open(args.result_file.replace('persection', section))) for section in SECTION_DIVISIONS]
        
        data = deepcopy(data_list[0])
        for item in data:
            del item['output']
        
        # combine output
        for item_id, item in enumerate(data):
            output = ""
            for section_data, section in zip(data_list, SECTION_DIVISIONS):
                section_test_key = '%s_%s' % ('output', section)
                item[section_test_key] = section_data[item_id]['output'] # do not use section tagger for persection generation 
                output = output + section_data[item_id]['output'] + '\n'
            item['output'] = output
            
        KEYS_TO_SPLIT = ['reference']
    else:
        data = json.load(open(args.result_file))
        KEYS_TO_SPLIT = ['reference', 'output']
    
    for item in data:
        for text_key in KEYS_TO_SPLIT:
            if text_key in item:
                text = item[text_key]
                detected_divisions = section_tagger.divide_note_by_metasections(text)
                for detected_division in detected_divisions:
                    label, _, _, start, _, end = detected_division
                    section_test_key = '%s_%s' % (text_key, label)
                    item[section_test_key] = text[start:end]
                    
                for section in SECTION_DIVISIONS:
                    section_test_key = '%s_%s' % (text_key, section)
                    if section_test_key not in item:
                        item[section_test_key] = ""
    
    json.dump(data, open(args.result_file, 'w'), indent=4)