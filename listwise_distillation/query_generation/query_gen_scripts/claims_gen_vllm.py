from datasets import load_dataset
import json
from tqdm import tqdm
import transformers
import torch
import random
from vllm import LLM, SamplingParams

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llm = LLM(model=model_id)
PARENT_DIR = 'listwise_distillation/query_generation'

beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'arguana', 'webis-touche2020', 'scidocs', 'climate-fever']

for dataset in beir_datasets:

    gpt_examples = []
    with open(f'{PARENT_DIR}/prompt_examples/gpt_examples_claims.json', 'r', encoding='utf-8') as input_pairs:
        gpt_examples = json.load(input_pairs)

    system_prompt = 'You are tasked with creating a concise and relevant claim based on a given passage.'
    user_prompt = '''Instructions: Read the target passage carefully to understand its main idea. Then, write a clear, specific claim that captures a key point from the passage, ensuring that it is directly addressed by the passage and can stand alone without additional context. The claim should be concise and under 20 words. Do not copy long phrases or sections directly from the passage. Respond only with the plain text claim.
    Examples:

    Target Passage: {}
    Claim: {}

    Target Passage: {}
    Claim: {}

    Target Passage: {}
    Claim: {}

    Now create a claim for the following passage:
    Target Passage: {}
    Claim: 
    '''

    keep_ids = set()
    with open(f'{PARENT_DIR}/chosen_ids/' + dataset + '_chosen_ids.txt', 'r', encoding='utf-8') as f:
        for line in f:
            keep_ids.add(line.strip())
    with open(f'{PARENT_DIR}/test_ids/' + dataset + '_test_ids.txt', 'r', encoding='utf-8') as f:
        for line in f:
            keep_ids.add(line.strip())

    ds = load_dataset("BeIR/" + dataset, "corpus")['corpus']; id_key='_id'
        
    passages_dict = {}

    for line in tqdm(ds):
        if str(line[id_key]) in keep_ids:
            passages_dict[str(line[id_key])] = (str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048])

    print("len(passages_list)", len(passages_dict))

    llm_inputs = []
    ids_list = []
    for id in passages_dict:
        ids_list.append(id)
        few_shot_examples = random.sample(gpt_examples, 3)
        
        example_target_passage_1 = few_shot_examples[0]['passage']
        example_query_1 = few_shot_examples[0]['query']
        example_target_passage_2 = few_shot_examples[1]['passage']
        example_query_2 = few_shot_examples[1]['query']
        example_target_passage_3 = few_shot_examples[2]['passage']
        example_query_3 = few_shot_examples[2]['query']
        
        passage_to_generate_query = passages_dict[id]
        target_passage = passage_to_generate_query
        
        llm_input = user_prompt.format(example_target_passage_1, example_query_1,
                                    example_target_passage_2, example_query_2,
                                    example_target_passage_3, example_query_3,
                                    target_passage)

        llm_input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_input},    
        ]

        llm_input = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(llm_input, tokenize=False, add_generation_template=True)
        llm_inputs.append(llm_input)

    index = 0
    with open('generated_queries/' + dataset + '_generated_queries_claims.tsv', 'w') as f:
        sampling_params = SamplingParams(skip_special_tokens=True, max_tokens=50, min_tokens=1, temperature=0.8)
        outputs = llm.generate(
            llm_inputs,
            sampling_params
        )
        for output in outputs:
            generated_text = output.outputs[0].text
            if '\n' in generated_text:
                generated_text = generated_text[generated_text.index('\n'):].strip()
            if '\n' in generated_text:
                generated_text = generated_text[:generated_text.index('\n')].strip()
            
            generated_text = generated_text.replace('\n', ' ').replace('\t', ' ')
            
            if (generated_text.startswith('"')) and (generated_text.endswith('"')):
                generated_text = generated_text[1:-1]

            f.write(ids_list[index] + '\t' + generated_text + '\n')
            index +=1



