import tensorflow as tf
from hparams import *

def masked_sparse_categorical_accuracy(y_true, y_pred, word2idx): 
    mask = tf.where(y_true == word2idx['<PAD>'], 0., 1.) # (B,T)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32)*mask)/tf.reduce_sum(mask)
    return accuracy
