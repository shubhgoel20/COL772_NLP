#Data handling
from torch.utils.data import Dataset
import json
import string
from datetime import datetime
import calendar
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import random
import nltk
from gensim.models import FastText

# nltk.download('punkt')
class ColDataset(Dataset):
    def __init__(self, root_dir, word_vec_model, train = True):
        self.root_dir = root_dir
        self.samples = self._load_samples(root_dir)
        self.train = train
        self.word_vec_model = word_vec_model

    def _load_samples(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Load each line as a JSON object
                json_obj = json.loads(line)
                data.append(json_obj)
        return data

    def __len__(self):
        return len(self.samples)
    
    def date_to_words(self, date_string):
        try:
            # Attempt to parse the date string using different formats
            date_obj = None
            for date_format in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%y", "%Y-%m-%y", "%d-%m-%y"]:
                try:
                    date_obj = datetime.strptime(date_string, date_format)
                    break  # Stop loop if parsing is successful
                except ValueError:
                    continue  # Try next format if parsing fails
            
            if date_obj is None:
                return "Invalid date format"
            
            # Get the day, month, and year from the datetime object
            day = date_obj.day
            month = date_obj.month
            year = date_obj.year
            
            # Get the name of the month from the calendar module
            month_name = calendar.month_name[month]
            
            # Create a string representing the date in words
            date_in_words = f"{day} {month_name} {year}"
            
            return date_in_words
        except ValueError:
            return "Invalid date format"


    def get_char_vector(self, token):
        vec = []
        for char in token:
            if char in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[char]))
        return vec
    
    def get_word_vector(self, token):
        #token has no spaces
        if token in self.word_vec_model.key_to_index:
            return [torch.tensor(self.word_vec_model[token])]
        
        token = self.remove_punctuation(token) #still has /
        
        #check for date
        temp = self.date_to_words(token)

        if temp != "Invalid date format":
            token = temp.split()
            return [torch.tensor(self.word_vec_model[token[0]]), torch.tensor(self.word_vec_model[token[1]]), torch.tensor(self.word_vec_model[token[2]])]

        new_tokens = token.split('/')
        vec = []
        for new_token in new_tokens:
            if len(new_token) < 1:
                continue
            if new_token in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[new_token]))
            else:
                vec.extend(self.get_char_vector(new_token))
        
        return vec

    def remove_punctuation(self,text):
        translator = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
        return text.translate(translator)
        
    def preprocess_question(self, question):
        # tokens = question.lower().split()
        tokens = nltk.word_tokenize(question.lower())
        question_vec = []
        for token in tokens:
            if len(token) < 1:
                continue
            question_vec.extend(self.get_word_vector(token))
        
        question_vec = torch.stack(question_vec,dim = 0)
        return question_vec
    
    def preprocess_table(self, table):
        columns = []
        for i,col in enumerate(table['cols']):
            # tokens = col.lower().split()
            tokens = nltk.word_tokenize(col.lower())
            col_vec = []
            for token in tokens:
                if(len(token) < 1):
                    continue
                col_vec.extend(self.get_word_vector(token))
            col_vec = torch.stack(col_vec, dim = 0)
            columns.append(col_vec)    

        return columns

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        table = sample['table']

        num_cols = len(table['cols'])
        question_vec = self.preprocess_question(question)
        columns = self.preprocess_table(table)
    
        tgt_col = 0
        for i,col in enumerate(sample['table']['cols']):
            if col == sample['label_col'][0]:
                tgt_col = i
                break

        tgt_col = torch.tensor(tgt_col)
        ques_len = question_vec.shape[0]

        return (question_vec, columns, ques_len, num_cols, tgt_col)

def col_collate_fn(batch):
    # Extract individual elements from each sample in the batch
    questions, columns, ques_len_list, num_cols_list, tgt_cols = zip(*batch)

    # Pad questions and tables
    padded_questions = pad_sequence(questions, batch_first=True)

    num_cols_tensor = torch.tensor(num_cols_list)
    tgt_cols_tensor = torch.stack(tgt_cols, dim=0)
    ques_len_tensor = torch.tensor(ques_len_list)
    # tgt_rows_tensor = torch.stack(tgt_rows, dim=0)

    # Return the collated batch
    return padded_questions, columns, ques_len_tensor, num_cols_tensor, tgt_cols_tensor

class RowDataset2(Dataset):
    def __init__(self, root_dir, word_vec_model, train = True, val = False, saved = False):
        self.root_dir = root_dir
        self.samples = self._load_samples(root_dir)
        self.train = train
        if((not val) and (not saved)):
            self.fasttext = self.learn_embeddings(self.samples)
        else:
            self.fasttext = FastText.load("fasttext_model2.bin")
        self.word_vec_model = word_vec_model

    def _load_samples(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Load each line as a JSON object
                json_obj = json.loads(line)
                data.append(json_obj)
        return data

    def learn_embeddings(self, samples):
        print("Learning embeddings")
        corpus = []
        for i,sample in enumerate(samples):
            question = sample['question'].lower()
            question = nltk.word_tokenize(question)
            for j,col in enumerate(sample['table']['cols']):
                col = nltk.word_tokenize(col.lower())
                temp = question+col
                for k, row in enumerate(sample['table']['rows']):
                    cell = nltk.word_tokenize(row[j].lower())
                    temp2 = temp+cell
                    corpus.append(temp2)

        model = FastText(sentences=corpus, vector_size=300, window=3, min_count=1, workers=4, sg=1)
        print("embeddings_learnt")
        model.save("fasttext_model2.bin")
        return model

    def __len__(self):
        return len(self.samples)
    
    def date_to_words(self, date_string):
        try:
            # Attempt to parse the date string using different formats
            date_obj = None
            for date_format in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%y", "%Y-%m-%y", "%d-%m-%y"]:
                try:
                    date_obj = datetime.strptime(date_string, date_format)
                    break  # Stop loop if parsing is successful
                except ValueError:
                    continue  # Try next format if parsing fails
            
            if date_obj is None:
                return "Invalid date format"
            
            # Get the day, month, and year from the datetime object
            day = date_obj.day
            month = date_obj.month
            year = date_obj.year
            
            # Get the name of the month from the calendar module
            month_name = calendar.month_name[month]
            
            # Create a string representing the date in words
            date_in_words = f"{day} {month_name} {year}"
            
            return date_in_words
        except ValueError:
            return "Invalid date format"


    def get_char_vector(self, token):
        vec = []
        for char in token:
            if char in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[char]))
        return vec
    
    def get_word_vector(self, token):
        #token has no spaces
        return [torch.tensor(self.fasttext.wv[token])]
        
        if token in self.word_vec_model.key_to_index:
            return [torch.tensor(self.word_vec_model[token])]
        
        token = self.remove_punctuation(token) #still has /
        
        #check for date
        temp = self.date_to_words(token)

        if temp != "Invalid date format":
            token = temp.split()
            return [torch.tensor(self.word_vec_model[token[0]]), torch.tensor(self.word_vec_model[token[1]]), torch.tensor(self.word_vec_model[token[2]])]

        new_tokens = token.split('/')
        vec = []
        for new_token in new_tokens:
            if len(new_token) < 1:
                continue
            if new_token in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[new_token]))
            else:
                vec.extend(self.get_char_vector(new_token))
        
        return vec

    def remove_punctuation(self,text):
        translator = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
        return text.translate(translator)
        
    def preprocess_question(self, question):
        # tokens = question.lower().split()
        tokens = nltk.word_tokenize(question.lower())
        question_vec = []
        for token in tokens:
            if len(token) < 1:
                continue
            question_vec.extend(self.get_word_vector(token))
        
        question_vec = torch.stack(question_vec,dim = 0)
        return question_vec
    
    def preprocess_table(self, table):
        columns = []
        for i,col in enumerate(table['cols']):
            # tokens = col.lower().split()
            tokens = nltk.word_tokenize(col.lower())
            col_vec = []
            for token in tokens:
                if(len(token) < 1):
                    continue
                col_vec.extend(self.get_word_vector(token))
            col_vec = torch.stack(col_vec, dim = 0)
            columns.append(col_vec)

        rows = []
        for row in table['rows']:
            row = " ".join(row)
            row = row.lower()
            tokens = nltk.word_tokenize(row)
            row_vec = []
            for token in tokens:
                row_vec.extend(self.get_word_vector(token))
            row_vec = torch.stack(row_vec, dim=0)
            rows.append(row_vec)   

        return columns, rows

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        table = sample['table']

        num_cols = len(table['cols'])
        num_rows = len(table['rows'])
        question_vec = self.preprocess_question(question)
        columns, rows = self.preprocess_table(table)
    
        tgt_col = 0
        for i,col in enumerate(sample['table']['cols']):
            if col == sample['label_col'][0]:
                tgt_col = i
                break

        tgt_col = torch.tensor(tgt_col)
        tgt_rows = torch.tensor(sample['label_row'][0], dtype = torch.long)
        ques_len = question_vec.shape[0]

        return (question_vec, columns, rows, ques_len, num_cols, num_rows, tgt_col, tgt_rows)

def row_collate_fn(batch):
    # Extract individual elements from each sample in the batch
    questions, columns, rows, ques_len_list, num_cols_list, num_rows, tgt_cols, tgt_rows = zip(*batch)

    # Pad questions and tables
    padded_questions = pad_sequence(questions, batch_first=True)

    num_cols_tensor = torch.tensor(num_cols_list)
    # num_rows = torch.tensor(num_rows)
    tgt_cols_tensor = torch.stack(tgt_cols, dim=0)
    ques_len_tensor = torch.tensor(ques_len_list)
    tgt_rows_tensor = torch.stack(tgt_rows, dim=0)

    # Return the collated batch
    return padded_questions, columns, rows, ques_len_tensor, num_cols_tensor, num_rows, tgt_cols_tensor, tgt_rows_tensor

class ValColDataset(Dataset):
    def __init__(self, root_dir, word_vec_model, train = True):
        self.root_dir = root_dir
        self.samples = self._load_samples(root_dir)
        self.train = train
        self.word_vec_model = word_vec_model

    def _load_samples(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Load each line as a JSON object
                json_obj = json.loads(line)
                data.append(json_obj)
        return data

    def __len__(self):
        return len(self.samples)
    
    def date_to_words(self, date_string):
        try:
            # Attempt to parse the date string using different formats
            date_obj = None
            for date_format in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%y", "%Y-%m-%y", "%d-%m-%y"]:
                try:
                    date_obj = datetime.strptime(date_string, date_format)
                    break  # Stop loop if parsing is successful
                except ValueError:
                    continue  # Try next format if parsing fails
            
            if date_obj is None:
                return "Invalid date format"
            
            # Get the day, month, and year from the datetime object
            day = date_obj.day
            month = date_obj.month
            year = date_obj.year
            
            # Get the name of the month from the calendar module
            month_name = calendar.month_name[month]
            
            # Create a string representing the date in words
            date_in_words = f"{day} {month_name} {year}"
            
            return date_in_words
        except ValueError:
            return "Invalid date format"


    def get_char_vector(self, token):
        vec = []
        for char in token:
            if char in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[char]))
        return vec
    
    def get_word_vector(self, token):
        #token has no spaces
        if token in self.word_vec_model.key_to_index:
            return [torch.tensor(self.word_vec_model[token])]
        
        token = self.remove_punctuation(token) #still has /
        
        #check for date
        temp = self.date_to_words(token)

        if temp != "Invalid date format":
            token = temp.split()
            return [torch.tensor(self.word_vec_model[token[0]]), torch.tensor(self.word_vec_model[token[1]]), torch.tensor(self.word_vec_model[token[2]])]

        new_tokens = token.split('/')
        vec = []
        for new_token in new_tokens:
            if len(new_token) < 1:
                continue
            if new_token in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[new_token]))
            else:
                vec.extend(self.get_char_vector(new_token))
        
        return vec

    def remove_punctuation(self,text):
        translator = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
        return text.translate(translator)
        
    def preprocess_question(self, question):
        # tokens = question.lower().split()
        tokens = nltk.word_tokenize(question.lower())
        question_vec = []
        for token in tokens:
            if len(token) < 1:
                continue
            question_vec.extend(self.get_word_vector(token))
        
        question_vec = torch.stack(question_vec,dim = 0)
        return question_vec
    
    def preprocess_table(self, table):
        columns = []
        for i,col in enumerate(table['cols']):
            # tokens = col.lower().split()
            tokens = nltk.word_tokenize(col.lower())
            col_vec = []
            for token in tokens:
                if(len(token) < 1):
                    continue
                col_vec.extend(self.get_word_vector(token))
            col_vec = torch.stack(col_vec, dim = 0)
            columns.append(col_vec)    

        return columns

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        table = sample['table']

        num_cols = len(table['cols'])
        question_vec = self.preprocess_question(question)
        columns = self.preprocess_table(table)
    
        ques_len = question_vec.shape[0]

        return (question_vec, columns, ques_len, num_cols, sample['qid'], sample['table']['cols'])

def val_col_collate_fn(batch):
    # Extract individual elements from each sample in the batch
    questions, columns, ques_len_list, num_cols_list, qid, col_labels = zip(*batch)

    # Pad questions and tables
    padded_questions = pad_sequence(questions, batch_first=True)

    num_cols_tensor = torch.tensor(num_cols_list)
    ques_len_tensor = torch.tensor(ques_len_list)
    # tgt_rows_tensor = torch.stack(tgt_rows, dim=0)

    # Return the collated batch
    return padded_questions, columns, ques_len_tensor, num_cols_tensor, qid, col_labels

class ValRowDataset2(Dataset):
    def __init__(self, root_dir, word_vec_model, train = True, val = False, saved = False):
        self.root_dir = root_dir
        self.samples = self._load_samples(root_dir)
        self.train = train
        if((not val) and (not saved)):
            self.fasttext = self.learn_embeddings(self.samples)
        else:
            self.fasttext = FastText.load("fasttext_model2.bin")
        self.word_vec_model = word_vec_model

    def _load_samples(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Load each line as a JSON object
                json_obj = json.loads(line)
                data.append(json_obj)
        return data

    def __len__(self):
        return len(self.samples)
    
    def date_to_words(self, date_string):
        try:
            # Attempt to parse the date string using different formats
            date_obj = None
            for date_format in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%y", "%Y-%m-%y", "%d-%m-%y"]:
                try:
                    date_obj = datetime.strptime(date_string, date_format)
                    break  # Stop loop if parsing is successful
                except ValueError:
                    continue  # Try next format if parsing fails
            
            if date_obj is None:
                return "Invalid date format"
            
            # Get the day, month, and year from the datetime object
            day = date_obj.day
            month = date_obj.month
            year = date_obj.year
            
            # Get the name of the month from the calendar module
            month_name = calendar.month_name[month]
            
            # Create a string representing the date in words
            date_in_words = f"{day} {month_name} {year}"
            
            return date_in_words
        except ValueError:
            return "Invalid date format"


    def get_char_vector(self, token):
        vec = []
        for char in token:
            if char in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[char]))
        return vec
    
    def get_word_vector(self, token):
        #token has no spaces
        return [torch.tensor(self.fasttext.wv[token])]
        
        if token in self.word_vec_model.key_to_index:
            return [torch.tensor(self.word_vec_model[token])]
        
        token = self.remove_punctuation(token) #still has /
        
        #check for date
        temp = self.date_to_words(token)

        if temp != "Invalid date format":
            token = temp.split()
            return [torch.tensor(self.word_vec_model[token[0]]), torch.tensor(self.word_vec_model[token[1]]), torch.tensor(self.word_vec_model[token[2]])]

        new_tokens = token.split('/')
        vec = []
        for new_token in new_tokens:
            if len(new_token) < 1:
                continue
            if new_token in self.word_vec_model.key_to_index:
                vec.append(torch.tensor(self.word_vec_model[new_token]))
            else:
                vec.extend(self.get_char_vector(new_token))
        
        return vec

    def remove_punctuation(self,text):
        translator = str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
        return text.translate(translator)
        
    def preprocess_question(self, question):
        # tokens = question.lower().split()
        tokens = nltk.word_tokenize(question.lower())
        question_vec = []
        for token in tokens:
            if len(token) < 1:
                continue
            question_vec.extend(self.get_word_vector(token))
        
        question_vec = torch.stack(question_vec,dim = 0)
        return question_vec
    
    def preprocess_table(self, table):
        columns = []
        for i,col in enumerate(table['cols']):
            # tokens = col.lower().split()
            tokens = nltk.word_tokenize(col.lower())
            col_vec = []
            for token in tokens:
                if(len(token) < 1):
                    continue
                col_vec.extend(self.get_word_vector(token))
            col_vec = torch.stack(col_vec, dim = 0)
            columns.append(col_vec)

        rows = []
        for row in table['rows']:
            row = " ".join(row)
            row = row.lower()
            tokens = nltk.word_tokenize(row)
            row_vec = []
            for token in tokens:
                row_vec.extend(self.get_word_vector(token))
            row_vec = torch.stack(row_vec, dim=0)
            rows.append(row_vec)   

        return columns, rows

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        table = sample['table']

        num_cols = len(table['cols'])
        num_rows = len(table['rows'])
        question_vec = self.preprocess_question(question)
        columns, rows = self.preprocess_table(table)
    
        ques_len = question_vec.shape[0]

        return (question_vec, columns, rows, ques_len, num_cols, num_rows)

def val_row_collate_fn(batch):
    # Extract individual elements from each sample in the batch
    questions, columns, rows, ques_len_list, num_cols_list, num_rows = zip(*batch)

    # Pad questions and tables
    padded_questions = pad_sequence(questions, batch_first=True)

    num_cols_tensor = torch.tensor(num_cols_list)
    # num_rows = torch.tensor(num_rows)
    ques_len_tensor = torch.tensor(ques_len_list)

    # Return the collated batch
    return padded_questions, columns, rows, ques_len_tensor, num_cols_tensor, num_rows




