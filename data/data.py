import numpy as np
import tensorflow as tf
import nltk
import pickle
from hparams import *
from data.preprocessor import data_split

# Read dictionary pkl file
def open_vocabulary():
    with open('vocabulary.pkl', 'rb') as fp:
        vocabulary = pickle.load(fp)
    return vocabulary

def get_word2idx():
    vocabulary = open_vocabulary()
    word2idx = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
    word2idx_update = dict(zip(vocabulary.keys(), range(4,VOCAB_SIZE)))
    word2idx.update(word2idx_update)
    return word2idx

def get_idx2word():
    word2idx = get_word2idx()
    idx2word = dict(zip( word2idx.values(),word2idx.keys()))
    return idx2word

def tokenize_wrapper():
    word2idx = get_word2idx()
    def tokenize(review):
        """
        Split reviews into tokens.
        """
        review = review.decode('utf-8')
        words_list = nltk.tokenize.word_tokenize(review)
        index_list = [word2idx['<SOS>']]
        for word in words_list:
            index = word2idx.get(word,word2idx['<UNK>'])
            index_list.append(index)
        index_list.append(word2idx['<EOS>'])
        return np.array(index_list).astype(np.int32)
    return tokenize

def map_fn(review):
    """
    Apply tokenize to elements and setting maximum review length to 1500.
    """
    x = tf.numpy_function(tokenize_wrapper(), inp = [review], Tout = tf.int32)
    x = tf.ensure_shape(x, [None])
    return x[:MAX_REVIEW_LEN]

def batch_map_fn(batch):
    """
    Creating inputs and targets by removing last and first element respectively.
    """
    x=tf.shape(batch) 
    inputs = batch[:,0:x[1]-1]
    outputs = batch[:,1:x[1]]
    return inputs, outputs

def make_dataset(reviews, shuffle,word2idx):
    dataset = tf.data.Dataset.from_tensor_slices(reviews)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = dataset.cardinality(), reshuffle_each_iteration=True)
    dataset = dataset.map(map_fn, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size = BATCH_SIZE, padding_values=word2idx['<PAD>'] , drop_remainder = True)
    dataset = dataset.map(batch_map_fn, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_val_dataset():
    word2idx = get_word2idx()
    reviews_train, reviews_val = data_split()
    dataset_train = make_dataset(reviews_train, shuffle=True, word2idx=word2idx) # (inputs, targets)
    dataset_val = make_dataset(reviews_val, shuffle=True, word2idx=word2idx) # (inputs, targets)
    return dataset_train, dataset_val
