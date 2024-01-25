import tensorflow as tf
from hparams import *
from data.data import get_word2idx

class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Sparse categorical crossentropy but padding isn't counted in equation.
    """
    def __init__(self, name='masked_sparse_categorical_crossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.word2idx = get_word2idx()

    def call(self, y_true, y_pred):     
        
        loss = self.scce(y_true, y_pred) # y_true(B,T) y_pred(B,T,C)
        mask = tf.where(y_true == self.word2idx['<PAD>'], 0., 1.) # (B,T)
        loss = tf.reduce_sum(loss * mask)/ tf.reduce_sum(mask) # ()
        return loss
    