import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense
from hparams import *

class LSTMBlock(tf.keras.layers.Layer):
    """
    Class that creates LSTM layers based on list of units
    """
    def __init__(self, name='lstm_block'):
        super(LSTMBlock, self).__init__(name=name)
        self.lstm_list=[]        
        for units in LSTM_UNITS:            
            self.lstm_list.append(LSTM(units, return_sequences=True))
                
    def call(self,x, training):
        for lstm in self.lstm_list:
            x = lstm(x, training=training)
        return x        

class LSTMModel(Model):
    """
    Class that takes an input and runs it through model layers
    """
    def __init__(self, name='food_review_generation'):
        super(LSTMModel, self).__init__(name=name)
        self.embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_OUTPUT_DIM)
        self.lstm_block = LSTMBlock()
        self.dense = Dense (DENSE_UNITS, activation='relu')
        self.output_layer = Dense(VOCAB_SIZE)
        
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.lstm_block(x, training)
        x = self.dense(x)
        x = self.output_layer(x)
        return x
    
    def model(self):
        x = Input((None,))
        return Model(inputs=x, outputs=self.call(x, True))