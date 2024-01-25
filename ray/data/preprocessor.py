import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from hparams import *

def data_split():
    path = 'dataset/Reviews.csv'
    reviews = pd.read_csv(path, usecols=['Text'])
    reviews = np.squeeze(reviews, axis = 1)
    reviews = np.unique(reviews)
    reviews_train, reviews_val = train_test_split(reviews, test_size=0.1,random_state=2409)
    return reviews_train, reviews_val

def vocabulary():
    reviews_train,_ = data_split()
    train_string = ' '.join(reviews_train)
    words_list = nltk.tokenize.word_tokenize(train_string)
    words_and_frequency = nltk.FreqDist(words_list)
    counts = list(words_and_frequency.values())
    words = list(words_and_frequency.keys())
    counts = np.array(counts)
    words =np.array(words)
    count_sort_ind = np.argsort(-counts)
    words = words[count_sort_ind]
    counts = counts[count_sort_ind]
    vocabulary = dict(zip(words, counts))
    return vocabulary