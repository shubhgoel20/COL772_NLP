from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import pickle
import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# model_checkpoint_path = 'model_results_flant5_v1/checkpoint-21840'
model_checkpoint_path = sys.argv[2]
subdirs = [d for d in os.listdir(model_checkpoint_path) if os.path.isdir(os.path.join(model_checkpoint_path, d))]
checkpoint_dirs = [d for d in subdirs if d.startswith('checkpoint-')]
sorted_checkpoints = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
latest_checkpoint = sorted_checkpoints[-1] if sorted_checkpoints else None
full_checkpoint_path = os.path.join(model_checkpoint_path, latest_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(full_checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(full_checkpoint_path)

directory = sys.argv[1]
dataset_eLife = load_dataset('json', data_files={
    'test': directory + '/eLife_test.jsonl'
})

dataset_PLOS = load_dataset('json', data_files={
    'test': directory + '/PLOS_test.jsonl'
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def generate_summary(sample):
    article = sample['article']
    segments = article.split('\n')
    prompt = [f"summarize : \nAbstract : {segments[0]}\nResults : {segments[-1]}"]
    input_ids = tokenizer(prompt, max_length=512, truncation=True, return_tensors='pt', padding='max_length')['input_ids'][0]
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    tokenized_output = model.generate(input_ids = input_ids, min_length=30, max_length=400)
    output = tokenizer.decode(tokenized_output[0], skip_special_tokens = True)
    return output

from tqdm import tqdm

if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

output_eLife = os.path.join(sys.argv[3],'elife.txt')
with open(output_eLife, 'w', encoding='utf-8') as file:
    for sample in tqdm(dataset_eLife['test']):
        summary = generate_summary(sample)
        file.write(summary + '\n')

output_PLOS = os.path.join(sys.argv[3],'plos.txt')
with open(output_PLOS, 'w', encoding='utf-8') as file:
    for sample in tqdm(dataset_PLOS['test']):
        summary = generate_summary(sample)
        file.write(summary + '\n')