import os
import pickle
import joblib
import json
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import argparse
import warnings
warnings.filterwarnings("ignore")

def remove_punctuation_and_digits(s):
    translator = str.maketrans(string.punctuation + string.digits, " " * (len(string.punctuation) + 10))
    return s.translate(translator)

def preprocess(X):
    processed_data = []
    for d in X:
        data = {}
        text = d['text']
        text = remove_punctuation_and_digits(text)
        text = text.split(' ')
        new_text = []
        for word in text:
            word = word.lower()
            if(len(word) < 1):
                continue
            else:
                new_text.append(word)
        data['text'] = new_text
        processed_data.append(data)

    return processed_data

def get_corpus(X, vocab, replace = True):
    corpus = []
    for d in X:
        sentence = ""
        for word in d['text']:
            if vocab.get(word) is not None:
                sentence+=word
            else:
                if(replace):
                    sentence+= "_unk_"
            sentence+=" "
        sentence = sentence[:-1]
        corpus.append(sentence)
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, help="path to json file")
    parser.add_argument("--save", type=str, default = "./", help="path to save model")
    parser.add_argument("--vectorizer", type=str, default="count", help="type of vectorizer")
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--replace", action = 'store_true')

    args = parser.parse_args()

    with open(args.test_data, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)

    
    class_id_path = os.path.join(args.save, "class_id.pkl")
    vocab_path = os.path.join(args.save, "vocab.pkl")
    words_path = os.path.join(args.save,"words.pkl")

    with open(class_id_path, 'rb') as f:
        class_id = pickle.load(f)

    with open(vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    
    with open(words_path, 'rb') as f:
        words = pickle.load(f)


    #Load the saved model
    model_path = os.path.join(args.save,"model.pkl")
    model = joblib.load(model_path)


    processed_test = preprocess(test_data)
    corpus = get_corpus(processed_test,words,args.replace)

    if args.vectorizer == 'count':
        vectorizer  = CountVectorizer(vocabulary=list(vocab_dict.keys()), ngram_range=(1,3), token_pattern= r"(?u)\b\w+\b")
        X_test = vectorizer.transform(corpus)

    if args.vectorizer == 'tdf':
        vectorizer = CountVectorizer(vocabulary=list(vocab_dict.keys()), token_pattern= r"(?u)\b\w+\b", ngram_range=(1,2))
        cnt_matrix = vectorizer.transform(corpus)
        tranformer = TfidfTransformer(sublinear_tf=True)
        X_test = tranformer.fit_transform(cnt_matrix)
    

    
    
    #Make predictions
    y_pred = model.predict(X_test)
    y_pred = list(y_pred)
    n = len(y_pred)

    with open(args.output, 'w') as file:
        for i,pred in enumerate(y_pred):
            for key,value in class_id.items():
                if(value == pred):
                    file.write(key)
                    break
            if( i != n-1):
                file.write("\n")
    
    print("Output file generated")
    

    

    


    

