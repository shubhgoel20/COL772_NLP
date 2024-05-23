import os
import sys
import time
import pickle
import joblib
import json
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import warnings
warnings.filterwarnings("ignore")

def remove_punctuation_and_digits(s):
    translator = str.maketrans(string.punctuation + string.digits, " " * (len(string.punctuation) + 10))
    return s.translate(translator)

def preprocess(X):
    processed_data = []
    id = 0
    class_id = {}
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
        data['langid'] = d['langid']
        processed_data.append(data)
        if class_id.get(d['langid']) is None:
            class_id[d['langid']] = id
            id+=1
    return processed_data, class_id

def get_corpus(X, class_id, freq = 2, replace = True):
    word_freq = {}
    for d in X:
        for word in d['text']:
            if word_freq.get(word) is None:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    
    y_train = []
    corpus = []
    words = {}
    for d in X:
        sentence = ""
        for word in d['text']:
            sentence+=word
            words[word] = 1
            sentence+=" "
        sentence = sentence[:-1]
        if len(sentence) < 1:
            continue
        corpus.append(sentence)
        y_train.append(class_id[d['langid']])
    y_train = np.array(y_train)
    return corpus, y_train, words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to json file")
    parser.add_argument("--save", type=str, default = "./", help="path to save model")
    parser.add_argument("--vectorizer", type=str, default="count", help="type of vectorizer")
    parser.add_argument("--model", type=str, default="NB", help="type of model")
    parser.add_argument("--freq", type=int, default=2, help="infrequent words frequency")
    parser.add_argument("--replace", action='store_true')


    args = parser.parse_args()

    start_time = time.time()

    train_data = []
    valid_data = []
    valid_new_data = []
    
    train_path = os.path.join(args.train_data,"train.json")
    valid_path = os.path.join(args.train_data,"valid.json")
    valid_new_path = os.path.join(args.train_data,"valid_new.json")

    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as fp:
            train_data = json.load(fp)
    else:
        print(train_path,"does not exist")
        sys.exit(1)

    if os.path.exists(valid_path):
        with open(valid_path, 'r', encoding='utf-8') as fp:
            valid_data = json.load(fp)
    else:
        print(valid_path,"does not exist")

    if os.path.exists(valid_new_path):
        with open(valid_new_path, 'r', encoding='utf-8') as fp:
            valid_new_data = json.load(fp)
    else:
        print(valid_new_path,"does not exist")
    
    train_data = train_data+valid_data+valid_new_data
    
    print("preprocessing:")
    processed_train, class_id = preprocess(train_data)
    corpus, y_train, words = get_corpus(processed_train,class_id, args.freq,args.replace)

    print("vectorizing:")
    if args.vectorizer == 'count':
        vectorizer  = CountVectorizer(token_pattern= r"(?u)\b\w+\b", ngram_range=(1,3))
        vectorizer.fit(corpus)
    
    if args.vectorizer == 'tdf':
        vectorizer  = TfidfVectorizer(token_pattern= r"(?u)\b\w+\b", ngram_range=(1,2) )
        vectorizer.fit(corpus)
    
    vocab_dict = {}
    for word in list(vectorizer.get_feature_names_out()):
        vocab_dict[word] = 1
    
    X_train = vectorizer.transform(corpus)
    
    print("Training:", args.model)

    if args.model == "NB":
        model = MultinomialNB(alpha=5e-5)
        model.fit(X_train,y_train)
    
    if args.model == "LR":
        model = LogisticRegression(random_state=0, n_jobs = 4, max_iter=300, multi_class='multinomial', verbose=1)
        model.fit(X_train,y_train)
    
    if args.model == "SVM":
        model = LinearSVC(random_state=0, verbose=1, tol= 1e-6)
        model.fit(X_train,y_train)
    
    if args.model == "SGD":
        model = SGDClassifier(random_state=0, verbose=1)
        model.fit(X_train,y_train)
    
    if args.model == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
        model.fit(X_train,y_train)

    #Save the class ids
    class_id_path = os.path.join(args.save,"class_id.pkl")
    with open(class_id_path, 'wb') as f:
        pickle.dump(class_id, f)
    
    #Save the vocabulary
    vocab_path = os.path.join(args.save,"vocab.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
    
    #Save the word_dict
    words_path = os.path.join(args.save,"words.pkl")
    with open(words_path, 'wb') as f:
        pickle.dump(words, f)

    #Save the model
    model_path = os.path.join(args.save,"model.pkl")
    joblib.dump(model, model_path)

    print("Training complete, model saved")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    

    


    

