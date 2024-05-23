import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import gensim
import gensim.downloader as api
import random
from data_handling import ColDataset, col_collate_fn
from models import ColBiLSTM2
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_file", type=str, help="train_file")
parser.add_argument("val_file", type=str, help="val_file")
args = parser.parse_args()


word_vec_model = api.load("fasttext-wiki-news-subwords-300")
root_dir_train = args.train_file
root_dir_val = args.val_file
train_dataset = ColDataset(root_dir= root_dir_train, word_vec_model=word_vec_model)
val_dataset = ColDataset(root_dir=root_dir_val, word_vec_model=word_vec_model)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=col_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=col_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ColBiLSTM2()
col_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=3e-3)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 3)
num_epochs = 10

model = model.to(device)
# sample = random.sample(range(train_dataset.__len__()), 10000)
#TRAINING LOOP
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    cnt = 0
    for data in tqdm(train_loader):
        question, columns, ques_len, num_cols, tgt_col = data
        question = question.to(device)

        temp = []
        for column in columns:
            column = [item.to(device) for item in column]
            temp.append(column)
        columns = temp

        ques_len = ques_len.to(device)
        num_cols = num_cols.to(device)
        tgt_col = tgt_col.to(device)
    
        col_logits = model(question, columns, ques_len, num_cols)

        loss = col_loss(col_logits[0], tgt_col[0])

        for i in range(1,tgt_col.shape[0]):
            loss+= col_loss(col_logits[i], tgt_col[i])

        loss = loss/tgt_col.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
        for i in range(tgt_col.shape[0]):
            pred_col = torch.argmax(col_logits[i]).item()
            if int(pred_col) == int(tgt_col[i].item()):
                cnt+=1      

        torch.cuda.empty_cache()
        
        
    N = len(train_loader)
    total_loss = total_loss/N
    acc = cnt/25000
    # scheduler.step(total_loss)
    if(epoch%1 == 0): 
        print(f"Epoch {epoch}/{num_epochs}, Train Col Loss: {total_loss:.4f}, Train Col Acc: {acc :.4f}")
        print("matched:", cnt)
        # print(pred_rows)

    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        cnt = 0
        for data in tqdm(val_loader):
            question, columns, ques_len, num_cols, tgt_col = data
            question = question.to(device)

            temp = []
            for column in columns:
                column = [item.to(device) for item in column]
                temp.append(column)
            columns = temp

            ques_len = ques_len.to(device)
            num_cols = num_cols.to(device)
            tgt_col = tgt_col.to(device)
        
            col_logits = model(question, columns, ques_len, num_cols)

            loss = col_loss(col_logits[0], tgt_col[0])

            for i in range(1,tgt_col.shape[0]):
                loss+= col_loss(col_logits[i], tgt_col[i])

            loss = loss/tgt_col.shape[0]
    
            total_loss += loss.item()
        
            for i in range(tgt_col.shape[0]):
                pred_col = torch.argmax(col_logits[i]).item()
                if int(pred_col) == int(tgt_col[i].item()):
                    cnt+=1      

            torch.cuda.empty_cache()
            
            
        N = len(val_loader)
        total_loss = total_loss/N
        acc = cnt/5000
        
        scheduler.step()
        
        if(epoch%1 == 0): 
            print(f"Epoch {epoch}/{num_epochs}, Val Col Loss: {total_loss:.4f}, Val Col Acc: {acc :.4f}")
            print("matched:", cnt)
            # print(pred_rows)
    torch.save(model,"ColPredictor.pth")