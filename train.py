import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from hparams import *
from nn.model import LSTMModel, CNNModel, transformer_model
from utils.utils import create_checkpoint_manager, load_checkpoint_if_exists
from data.data import train_val_dataset
from utils.train_helpers import train

if 'lstm' in MODEL_NAME:
    model = LSTMModel()
elif 'cnn' in MODEL_NAME:
    model = CNNModel()
elif 'transformer' in MODEL_NAME:
    model = transformer_model()
else:
    raise NotImplementedError(f'uuuuuh model {MODEL_NAME} type unknown!!!')

lr_schedule = ExponentialDecay(initial_learning_rate=INITIAL_LEARNING_RATE,
                               decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=False, name=None)
optimizer = Adam(lr_schedule)
manager = create_checkpoint_manager(f'checkpoints/{MODEL_NAME}', model, optimizer)
load_checkpoint_if_exists(manager)

train_summary_writer = tf.summary.create_file_writer(f'tb_logs/{MODEL_NAME}/train')
val_summary_writer = tf.summary.create_file_writer(f'tb_logs/{MODEL_NAME}/val')

dataset_train, dataset_val = train_val_dataset()

train(dataset_train=dataset_train, dataset_val=dataset_val, train_summary_writer=train_summary_writer,
     val_summary_writer=val_summary_writer, manager=manager, model=model, optimizer=optimizer)
