import streamlit as st

import tensorflow as tf
import os, re, pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt

economy_files = os.listdir("./Corpus/Economy")
economy = [pd.read_csv(os.path.join("./Corpus/Economy", file), names=["Text"]) for file in economy_files]

def clean_text(text):
    no_underscore = re.sub(r'_', ' ', text)
    lines = no_underscore.split("\n")
    filtered_lines = [re.sub(r'[^\u0600-\u06FF ]', '', line) for line in lines if line != ""]
    filtered =  '\n'.join(filtered_lines);
    no_diacritics = re.sub(r'[^\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z 0-9]', '', filtered)
    no_punctuations = re.sub(r'،؛؟«»!', '', no_diacritics)
    clean = re.sub(r'(.)\1+', r'\1', no_punctuations) # remove repeated characters
    return clean.lstrip().rstrip()


cleaned = [clean_text(corp['Text'].values[0]) for corp in economy[:50]]
tokenizer = Tokenizer()
corpus = [string.lower().split("\n")[0] for string in cleaned]
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index.items()) + 1

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
x, labels = input_sequences[:,:-1],input_sequences[:,-1]

y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

loaded_model = load_model('model/predictor_model.h5')
loaded_history = pickle.load(open("model/predictor_history.p", "rb"))

def find_word(vocab, index): return list(vocab)[index-1]

def last_string(string):
    tokens = string.split(" ")
    return tokens[len(tokens)-1]

def most_prob_words(predicted, string, vocab, n_words=3):
    sorted_list = predicted.copy()
    sorted_list.sort(reverse=True)
    index_list = [predicted.index(val) for val in sorted_list]
    probable_words = []
    c, i = 0, 0
    while c < n_words and i < len(index_list):
        word = find_word(vocab, index_list[i])
        if word.startswith(string):
            probable_words.append(word)
            c += 1
        i += 1
    return probable_words


def predict(seed_text):
    
    string = last_string(seed_text)
    vocab = tokenizer.word_index

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = loaded_model.predict(token_list, verbose=0)
    probable_words = most_prob_words(predicted[0].tolist(), string, vocab, n_words=5)
    for i in range(len(probable_words)):
        st.write(probable_words[i]) 
