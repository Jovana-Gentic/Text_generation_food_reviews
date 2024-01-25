from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from hparams import *
from nn.model import LSTMModel
from utils.utils import create_checkpoint_manager, load_checkpoint_if_exists
from utils.generate_helpers import generate_review

model = LSTMModel()
lr_schedule = ExponentialDecay(initial_learning_rate=INITIAL_LEARNING_RATE,
                               decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=False, name=None)
optimizer = Adam(lr_schedule)
manager = create_checkpoint_manager(f'checkpoints/{MODEL_NAME}', model, optimizer)
load_checkpoint_if_exists(manager)

generate_review(model, 'I really liked this tea', temperature=0.7)