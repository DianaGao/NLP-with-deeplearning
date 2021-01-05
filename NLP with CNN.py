from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score


#Word Vectors is refer to Standford pre-trained glove vectors

#Step one: Tokenizing (String split)
#convert a phrase into a list of strings which each element is called a token

#Step two: Pad sequences (rectangular output)
#By adding 0s to seuqneces that shorter than the other, it ensure all sequneces share the same length as the longest one.

#Configuration
MAX_SEQUENCE_LENGTH = 100 # Find the maximum length of a comment and set it up here
MAX_VOCAB_SIZE = 20000 # Rule of thumb: native English speaker knows apprximately 20,000 words
EMBEDDING_DIM = 100 # Embedding dimension
VALIDATION_SPLIT = 0.2 #Testc data/Total dataset
BATCH_SIZE = 128 #Number of data in one btach
EPOCHS = 10 #Numebr of times passing dataset into neural network

print('Loading word vectors...')
word2vec = {}
with open(os.path.join('D:/ML/NLP in deeplearning/Global Vectors for words representation/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding="utf8") as f:
    #file format is like: word vec[0] vec[1] vec[2] ...
    #Compudation complexity will increase as dimension increases
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype = 'float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

print('Loading in comments...')

train = pd.read_csv("train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE) # keep number of X more frequent words
tokenizer.fit_on_texts(sentences) # assign index based word frequency, higher frewuency lower index#
sequences = tokenizer.texts_to_sequences(sentences)

print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print("median sequence length:", s[len(s) // 2])

print("max word index:", max(max(seq) for seq in sequences if len(seq) > 0))

word2idx = tokenizer.word_index #convert word into integer
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

print('Filling pre-trained embeddings...')

num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1) #Index is +1 in Python
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            #words not found in embedding index will be all zeros. Dictionary might do not contacins the word we have
            embedding_matrix[i] = embedding_vector
