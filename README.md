# DocLens 🔍

<p align="left">
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://arxiv.org/abs/2311.09581"><img src="https://img.shields.io/badge/arXiv-2311.09581-b31b1b.svg"></a>
</p>

Code for "DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation" ([Arxiv](https://arxiv.org/abs/2311.09581))

If you find our paper or code useful, please cite the paper:
```
@misc{xie2024doclens,
      title={DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation}, 
      author={Yiqing Xie and Sheng Zhang and Hao Cheng and Pengfei Liu and Zelalem Gero and Cliff Wong and Tristan Naumann and Hoifung Poon and Carolyn Rose},
      year={2024},
      eprint={2311.09581},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

&nbsp;
# Data
To evaluate with DocLens, you will need two json files: 
* (1) a file with the input and reference, which should be put under `data/`, and
* (2) a file with the generated text, which should be put under `results/`

The file with the input and reference is a list of dicts. Each dict represents a test example and is in the following format:
```
{
    "example_id": the id of this example,
    "input": the input text,
    "reference": the reference output # Optional, required for claim recall/precision evaluation
}
```
Note that the `reference` key is required for the claim recall/precision evaluation, but is not required for citation recall/precision evaluation.

The file with the generated text is also a list of dicts with a similar format:
```
{
    "example_id": the id of this example,
    "input": the input text,
    "output": the system output 
}
```


&nbsp;
# Evaluation with DocLens
We provide the code to compute **claim recall**, **claim precision**, **citation recall**, and **citation precision**.


## Claim Generation
To evaluate **claim recall** and **claim precision**, we will need to first generate the subclaims for the reference and outputs by running:
```
bash scripts/eval_general_claim_generation.sh $SAVENAME $REFERENCE $PROMPT_FILE
```
`$SAVENAME` is the name of the file for generated text without the '.json' file extension (e.g., if your file is results/generation.json, we have $SAVENAME="generation").
`$REFERENCE` is the name of the file with the input and reference without the '.json' file extension (e.g., if your file is data/reference.json, we have $REFERENCE="reference").
`$PROMPT_FILE` is the prompt for claim extraction. We provide a simple prompt template in `claim_evaluation/prompts/general_subclaim_generation.json`. You can also create your own prompt file.


&nbsp;
## Claim Recall and Claim Precision Computation
After generating the claims, we can compute **claim recall** and **claim precision**. You can use the GPT-4 evaluator by running:
```
bash scripts/eval_general_api_claim_entailment.sh $SAVENAME $REFERENCE $PROMPT_FILE
```
We have $PROMPT_FILE="claim_evaluation/prompts/general_claim_entail.json" by default

You can also use the Mistral or TRUE evaluators:
```
bash scripts/eval_general_model_claim_entailment.sh $SAVENAME $REFERENCE $EVAL_MODEL $PROMPT_FILE
```
You can choose the evaluator model by setting `$EVAL_MODEL=TRUE` or `$EVAL_MODEL=Mistral`. If you want to use Mistral for evaluation, you can also specify the $PROMPT_FILE, which is by default `claim_evaluation/prompts/general_claim_entail_Mistral.json`


&nbsp;
## Citation Recall and Citation Precision Computation
The computation of **citation recall** and **citation precision** do not need reference.
You can use GPT-4 to compute citation recall and precision:
```
bash scripts/eval_general_api_citation.sh $SAVENAME $PROMPT_FILE
```
We have `$PROMPT_FILE="citation_evaluation/prompts/general_citation_entail.json"` by default.

You can also use the Mistral or TRUE evaluators:
```
bash scripts/eval_general_model_citation.sh $SAVENAME $EVAL_MODEL $PROMPT_FILE
```
We have `$PROMPT_FILE="citation_evaluation/prompts/general_citation_entail_Mistral.json"` by default.


&nbsp;
## Aggregate Scores
The scores of all examples can be aggregated by `aggregate_scores.py`. For example:
```
python aggregate_scores.py --result_file results/${SAVENAME}.json \
    --eval_claim_recall \       # compute claim recall
    --eval_claim_precision \    # compute claim precision
    --eval_citations \          # compute citation recall or citation precision
    --eval_model GPT            # can also be Mistral or TRUE, depend on the evaluator model you used
```


&nbsp;
# Reproduce the Results in our Paper
Here are the instructions for reproducing the results on ACI-BENCH (note generation), MIMIC (report summarization), and MeQSum (question summarization) in our paper.

## Data
We provide the preprocessed datafiles as follows:
```
data
├── ACI-Bench-TestSet-1_clean.claim_min1max30.json  # data of ACI-BENCH-test1 with generated reference claims
├── ACI-Bench-TestSet-1_clean.json                  # data of ACI-BENCH-test1
├── meqsum-test_clean.json                          # data of MeQSum-test
├── mimic-sampled200_clean.json                     # data of MIMIC (the 200 test examples sampled by proportion of different splits)
└── mimic-sampled200_clean.claim_min1max30.json     # data of MIMIC (200 samples) with generated reference claims
```
The `.claim_min1max30.json` files contain the reference subclaims we generated.


&nbsp;
## Run Medical Text Generation
To run text generation, you'll need to call the `run.py` file. This will follow the instructions in the prompt file and generate a piece of text based on the `input` text of each example.

We provide several example scripts unser `scripts/` named `run_DATASET.sh` or `run_DATASET_0shot.sh`. For example, 
```
bash scripts/run_mimic.sh $CONFIG_FILE
```
We provide several example config files under `configs/`
Note that for ACI-BENCH, we provide scripts for both full-note generation (e.g., `scripts/run_acibench_full.sh`) and per-section generation (e.g., `scripts/run_acibench_persection.sh`).

&nbsp;
## Evaluation with DocLens
The scripts for evaluation are similar to evaluating your own text. For example:
```
bash scripts/eval_mimic_api.sh $SAVENAME
```
and 
```
bash scripts/eval_mimic_model.sh $SAVENAME $EVAL_MODEL
```

