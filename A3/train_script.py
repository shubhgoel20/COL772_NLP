from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import pickle
import os
import sys

directory = sys.argv[1]

dataset_eLife = load_dataset('json', data_files={
    'train': directory + '/eLife_train.jsonl',
    'val': directory + '/eLife_val.jsonl'
})

dataset_PLOS = load_dataset('json', data_files={
    'train': directory + '/PLOS_train.jsonl',
    'val': directory + '/PLOS_val.jsonl'
})

dataset = DatasetDict()

for split in dataset_eLife.keys():
    dataset[split] = concatenate_datasets(
        [dataset_eLife[split], dataset_PLOS[split]]
    )
    dataset[split] = dataset[split].shuffle()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

# from transformers import AutoTokenizer, BioGptForCausalLM
# tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
# model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')

cache_file = 'processed_dataset.pkl'

def preprocess(dataset):
    processed_data = {'input_ids':[], 'labels':[]}
    for article, summary in zip(dataset['article'],dataset['lay_summary']):
        segments = article.split('\n')
        input_text = f"summarize : \nAbstract : {segments[0]}\nResults : {segments[-1]}"
        # can change max_length = 1024 when using bioGPT
        input_ids = tokenizer(input_text, max_length=512, truncation=True, return_tensors='pt', padding='max_length')['input_ids'][0]
        label_ids = tokenizer(summary, max_length=512, truncation=True, return_tensors='pt' ,padding='max_length')['input_ids'][0]
        
        processed_data['input_ids'].append(input_ids)
        processed_data['labels'].append(label_ids)
    return processed_data

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)
else:
    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.remove_columns(['headings','keywords','id','article','lay_summary'])
    
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)

print(dataset)
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r = 32,
    lora_alpha = 32,
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = TaskType.SEQ_2_SEQ_LM,
    target_modules='all-linear'
)

model = get_peft_model(model, peft_config=lora_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from transformers import TrainingArguments, Trainer

train_dataset = dataset['train']
val_dataset = dataset['val']

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=8)

batch_size = 8

training_args = TrainingArguments(
    output_dir=sys.argv[2],
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_gpu_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True,
    no_cuda=False
)

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset
)

trainer.train()