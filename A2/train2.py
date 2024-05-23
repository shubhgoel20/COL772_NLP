import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import gensim
import gensim.downloader as api
import random
from data_handling import RowDataset2, row_collate_fn
from models import RowBiLSTM3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_file", type=str, help="train_file")
args = parser.parse_args()

# word_vec_model = api.load("fasttext-wiki-news-subwords-300")
root_dir = args.train_file

train_dataset = RowDataset2(root_dir=root_dir, word_vec_model= None, val = False, saved = False)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn= row_collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = RowBiLSTM3()
# row_loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(5).to(device))
row_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)
num_epochs = 20

model = model.to(device)

#TRAINING LOOP
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    cnt = 0
    for data in tqdm(train_loader):
        question, columns, batch_rows, ques_len, num_cols, num_rows, tgt_col, tgt_rows = data
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
        # num_rows = num_rows.to(device)

        tgt_col = tgt_col.to(device)
        tgt_rows = tgt_rows.to(device)

        pred_row = model(question, columns, batch_rows, ques_len, num_cols, num_rows)

        r_loss = row_loss(pred_row[0], tgt_rows[0])

        for i in range(1,tgt_rows.shape[0]):
            r_loss+= row_loss(pred_row[i], tgt_rows[i])

        r_loss = r_loss/tgt_col.shape[0]

        
        loss = r_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for i in range(tgt_rows.shape[0]):
            pred = torch.argmax(pred_row[i]).item()
            if int(pred) == int(tgt_rows[i].item()):
                cnt+=1  
    
        torch.cuda.empty_cache()
        
    N = len(train_loader)
    total_loss = total_loss/N
    acc = cnt/25000

    if(epoch%1 == 0): 
        print(f"Epoch {epoch}/{num_epochs}, Train Row Loss: {total_loss:.4f}")
        print("acc: ", acc)
    torch.save(model, "RowPredictor.pth")
        