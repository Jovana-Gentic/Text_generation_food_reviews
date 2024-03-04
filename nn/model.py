import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.activations import relu
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


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 max_length,
                 initializer="glorot_uniform",
                 seq_axis=1,
                 **kwargs):
        super().__init__(**kwargs)
        if max_length is None:
            raise ValueError(
                "`max_length` must be an Integer, not `None`."
            )
        self._max_length = max_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._seq_axis = seq_axis

    def get_config(self):
        config = {
            "max_length": self._max_length,
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "seq_axis": self._seq_axis,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dimension_list = input_shape
        width = dimension_list[-1]
        weight_sequence_length = self._max_length

        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=[weight_sequence_length, width],
            initializer=self._initializer)

        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[self._seq_axis]
        position_embeddings = self._position_embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in inputs.get_shape().as_list()]
        new_shape[self._seq_axis] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)  
      
def transformer_block(inputs):
    x = MultiHeadAttention(TRANSFORMER_HEADS, TRANSFORMER_CHANNELS//TRANSFORMER_HEADS,
                           TRANSFORMER_CHANNELS//TRANSFORMER_HEADS)(inputs, inputs, inputs, use_causal_mask=True)
    inputs = LayerNormalization(axis=-1, epsilon=1e-6)(inputs + x)
    x = Dense(TRANSFORMER_CHANNELS*4, activation='relu')(inputs)
    x = Dense(TRANSFORMER_CHANNELS)(x)
    inputs = LayerNormalization(axis=-1, epsilon=1e-6)(inputs + x)
    return inputs

def transformer_model():
    position_embedding = PositionEmbedding(max_length=MAX_REVIEW_LEN)
    inputs = Input((None,)) 
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_OUTPUT_DIM)(inputs)
    x = x+position_embedding(x)
    for block in range(TRANSFORMER_BLOCK_NUMBER):
        x = transformer_block(x)
    outputs = Dense(VOCAB_SIZE)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class ResidualBlock(tf.keras.layers.Layer):
    """
    Creates one residual block with two convs.
    """
    def __init__(self, conv_config, name='residual_block'):
        super(ResidualBlock, self).__init__(name=name)
        self.conv_list=[]
        for filters, kernel_size, dilation_rate in conv_config:
            self.conv_list.append(Conv1D(filters, kernel_size, padding=PADDING,
                                         activation=None, dilation_rate=dilation_rate))

        self.filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.dense = Conv1D(self.filters, 1, padding=PADDING, activation=None, dilation_rate=1)
        
        super(ResidualBlock, self).build(input_shape=input_shape)
        
    def call(self, inputs):
        x = inputs
        for conv in self.conv_list:
            y = relu(x)
            x = conv(y)

        if inputs.shape != x.shape:
            inputs = self.dense(inputs)
            
        return inputs + x
    

class CNNModel(Model):

    def __init__(self, name='food_review_generation'):
        super(CNNModel, self).__init__(name=name)
        self.embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_OUTPUT_DIM)
        self.conv1D = Conv1D(CONV_FILTERS, KERNEL_SIZE, padding = PADDING, name = 'conv1')
        
        self.block_list=[]
        for block, conv_config in enumerate(CONV_CONFIGS):
            self.residual_block = ResidualBlock(conv_config, name=f'res_block{block}')
            self.block_list.append(self.residual_block)
        
        self.dense = Dense(DENSE_UNITS, activation='relu')
        self.output_layer = Dense(VOCAB_SIZE)
        
    def call(self, inputs):
        x = self.embedding(inputs)    
        x = self.conv1D(x)
        for block in self.block_list:
            x = block(x)
            
        x = relu(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x
    
    def model(self):
        x = Input((None,))
        return Model(inputs=x, outputs=self.call(x)) 
