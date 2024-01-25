import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from hparams import *
from data.data import get_word2idx, get_idx2word

def tokenize_test_string(test_string, word2idx):
    words_list = nltk.tokenize.word_tokenize(test_string)
    index_list = [word2idx['<SOS>']]
    for word in words_list:
        index = word2idx.get(word,word2idx['<UNK>'])
        index_list.append(index)
    return np.array(index_list).astype(np.int32)

def get_test_tokens_wrapper(model):
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.float32)])
    def get_test_tokens(x, temperature): # (B,T)
        logits = model(x, training=False)[:,-1:,:] # (B,1,C) 
        rand_uniform = tf.random.uniform(tf.shape(logits), minval=1e-5, maxval=1. - 1e-5) # (B,1,C)
        gumbel_noise = -tf.math.log(-tf.math.log(rand_uniform)) # (B,1,C)
        y_pred_sampling = tf.argmax(logits / temperature + gumbel_noise, axis=-1, output_type=tf.int32) # (B,1)
        x = tf.concat([x, y_pred_sampling], axis=1) # (B,T+1)
        return x
    return get_test_tokens

def detokenize_generated_review(x, idx2word):
    token_list_x = []
    for token in x:
        word = idx2word[token]
        token_list_x.append(word)
    return TreebankWordDetokenizer().detokenize(token_list_x)

def generate_review(model, x='', temperature=1.):
    word2idx = get_word2idx()
    idx2word = get_idx2word()
    temperature=tf.convert_to_tensor(temperature)
    x = tf.convert_to_tensor(tokenize_test_string(x, word2idx))[tf.newaxis,:] # (1, T)
    get_test_tokens = get_test_tokens_wrapper(model)
    for step in range(GENERATED_REVIEW_MAX_LEN):
        x = get_test_tokens(x, temperature) # (1, T)
        if x[0, -1]==word2idx['<EOS>']:
            break
    x = x[0,:].numpy() # (T,)
    return detokenize_generated_review(x, idx2word)