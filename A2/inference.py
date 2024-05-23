import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import gensim
import gensim.downloader as api
import random
from data_handling import ValColDataset, val_col_collate_fn, ValRowDataset2, val_row_collate_fn
import json
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("test_file", type=str, help="train_file")
parser.add_argument("pred_file", type=str, help="val_file")
args = parser.parse_args()


word_vec_model = api.load("fasttext-wiki-news-subwords-300")
root_dir = args.test_file


batch_size = 1

col_val_dataset = ValColDataset(root_dir=root_dir, word_vec_model=word_vec_model)
col_val_loader = DataLoader(col_val_dataset, batch_size=batch_size, collate_fn=val_col_collate_fn)
row_val_dataset = ValRowDataset2(root_dir=root_dir, word_vec_model= None, val = True, saved = True)
row_val_loader = DataLoader(row_val_dataset, batch_size=batch_size, collate_fn=val_row_collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

col_predictor = torch.load("ColPredictor.pth")
row_predictor = torch.load("RowPredictor.pth")

col_predictor = col_predictor.to(device)
row_predictor = row_predictor.to(device)

col_predictor.eval()
row_predictor.eval()


zipped_loaders = zip(col_val_loader, row_val_loader)

predictions = []

for data1,data2 in tqdm(zipped_loaders):
    question, columns, ques_len, num_cols, qid, column_labels = data1
    question = question.to(device)

    temp = []
    for column in columns:
        column = [item.to(device) for item in column]
        temp.append(column)
    columns = temp

    ques_len = ques_len.to(device)
    num_cols = num_cols.to(device)
    
    col_logits = col_predictor(question, columns, ques_len, num_cols)

    ###########################################################################################

    question, columns, batch_rows, ques_len, num_cols, num_rows = data2
    question = question.to(device)

    temp = []
    for column in columns:
        column = [item.to(device) for item in column]
        temp.append(column)
    columns = temp

    temp2 = []
    for rows in batch_rows:
        rows = [row.to(device) for row in rows]
        temp2.append(rows)
    batch_rows = temp2


    ques_len = ques_len.to(device)

    num_cols = num_cols.to(device)

    pred_row = row_predictor(question, columns, batch_rows, ques_len, num_cols, num_rows)
    
    #################################################################################################

    
    col_logits = col_logits[0]
    pred_col = column_labels[0][torch.argmax(col_logits).item()]
    pred_rows = [torch.argmax(pred_row[0]).item()]
    
    prediction = {}

    prediction['qid'] = qid[0]
    prediction['label_col'] = [pred_col]
    prediction['label_row'] = pred_rows
    prediction['label_cell'] = []
    for row in pred_rows:
        prediction['label_cell'].append([row,pred_col])
    predictions.append(prediction)

# Path to the JSONL file
file_path = args.pred_file

# Write the list of dictionaries to the JSONL file
with open(file_path, "w") as jsonl_file:
    for item in predictions:
        json.dump(item, jsonl_file)  # Write dictionary to file
        jsonl_file.write('\n')  # Add newline after each JSON object



