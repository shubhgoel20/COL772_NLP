import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class RowBiLSTM3(nn.Module):    
    def __init__(self, inp_channels = 300, d_model = 300):
        super(RowBiLSTM3, self).__init__()

        self.lstm_ques = torch.nn.LSTM(input_size= inp_channels, hidden_size= d_model, num_layers = 1, 
                                        bidirectional= True, batch_first = True)
        self.lstm_row = torch.nn.LSTM(input_size= inp_channels, hidden_size= d_model, num_layers = 1, 
                                        bidirectional= True, batch_first = True)

        self.inp_channels = inp_channels
        self.d_model = d_model
    
    def forward(self, question, columns, batch_rows_init, ques_len, num_cols, num_rows):
        # print(question.device)
        bs = question.shape[0]
        batch_idx = torch.arange(bs,device = question.device, dtype = torch.long)
        #question dim = bs, seq_len, inp_channels
        question,_ = self.lstm_ques(question) #bs, seq_len, inp_channels
        ques_len = ques_len - 1
        ques_embed_forward = question[batch_idx,ques_len,:self.d_model] #last lstm cell's output
        ques_embed_reverse = question[:,0,self.d_model:]
        ques_embed = torch.cat([ques_embed_forward,ques_embed_reverse], dim=1).to(question.device)
        # print(ques_embed.device)

        #TABLE PROCESSING
        #table dim = bs, seq_len, inp_channels
        batch_rows = []
        batch_row_lens = []
        for i in range(bs):
            col = torch.cat(columns[i], dim=0) #columns[i] = list of col tokens
            rows = batch_rows_init[i] #list of row tokens
            row_embed = []
            # rows2 = []
            row_lens = []
            for row in rows:
                row = torch.cat([col,row], dim=0)
                row_lens.append(row.shape[0])
                batch_rows.append(row)
            
            # print(len(rows2))
            # print(rows2[0].shape)
            row_lens = torch.tensor(row_lens, device=question.device, dtype= torch.long)
            batch_row_lens.append(row_lens)
            # rows2 = pad_sequence(rows2, batch_first=True)
            # # print(rows2.shape)
            # rows2,_ = self.lstm_row(rows2)

            # indices = torch.arange(rows2.shape[0], device = question.device, dtype = torch.long)
            # row_embed_forward = rows2[indices,row_lens-1,:self.d_model]
            # row_embed_reverse = rows2[:,0,self.d_model:]

            # row_embed = torch.cat([row_embed_forward,row_embed_reverse], dim=1)
            # # row_embed = torch.stack(row_embed, dim=0)
            # row_embeddings.append(row_embed)
        batch_rows = pad_sequence(batch_rows, batch_first=True)
        batch_rows,_ = self.lstm_row(batch_rows)

        batch_rows = torch.split(batch_rows, num_rows)
    
        logits = []
        for i,batch in enumerate(batch_rows):
            row_lens = batch_row_lens[i]
            indices = torch.arange(batch.shape[0], device = question.device, dtype = torch.long)
            row_embed_forward = batch[indices,row_lens-1,:self.d_model]
            row_embed_reverse = batch[:,0,self.d_model:]

            row_embed = torch.cat([row_embed_forward,row_embed_reverse], dim=1)

            logit = torch.matmul(ques_embed[i], torch.transpose(row_embed, 0,1))
            logit = logit.view(-1)
            logits.append(logit)
        # logits = torch.cat(logits,dim=0)
        # rows = rows.view(-1)
        # logits = F.sigmoid(logits)
        return logits


class ColBiLSTM2(nn.Module):    
    def __init__(self, inp_channels = 300, d_model = 300):
        super(ColBiLSTM2, self).__init__()

        self.lstm_ques = torch.nn.LSTM(input_size= inp_channels, hidden_size= d_model, num_layers = 2, 
                                        bidirectional= True, batch_first = True)
        self.lstm_col = torch.nn.LSTM(input_size= inp_channels, hidden_size= d_model, num_layers = 2, 
                                        bidirectional= True, batch_first = True)
        self.mlp = torch.nn.Linear(d_model, 2*d_model)

        self.inp_channels = inp_channels
        self.d_model = d_model
    
    def forward(self, question, columns, ques_len, num_cols):
        bs = question.shape[0]
        batch_idx = torch.arange(bs,device = question.device, dtype = torch.long)
        #question dim = bs, seq_len, inp_channels
        question,_ = self.lstm_ques(question) #bs, seq_len, inp_channels
        ques_len = ques_len - 1
        ques_embed_forward = question[batch_idx,ques_len,:self.d_model] #last lstm cell's output
        ques_embed_reverse = question[:,0,self.d_model:]
        ques_embed = torch.cat([ques_embed_forward,ques_embed_reverse], dim=1)

        #TABLE PROCESSING
        #table dim = bs, seq_len, inp_channels
        assert len(columns) == bs
        cols = []
        for i in range(bs):
            col = columns[i] #list of col tokens
            col_vec = []
            for j in col:        
                j_len = j.shape[0]
                j,_ = self.lstm_col(j)
                j_embed_forward = j[j_len-1,:self.d_model]
                j_embed_reverse = j[0,self.d_model:]
                col_vec.append(torch.cat([j_embed_forward, j_embed_reverse], dim=0))
            col_vec = torch.stack(col_vec, dim = 0)
            assert col_vec.shape[0] == num_cols[i]
            cols.append(col_vec)
            # table = table.permute(1,0,2)
        
        col_logits = []
        for i in range(bs):
            col_embed = cols[i]
            assert col_embed.shape[0] == num_cols[i]
            logits = torch.matmul(ques_embed[i], torch.transpose(col_embed, 0,1))
            logits = logits.view(-1)
            col_logits.append(logits)

        return col_logits

