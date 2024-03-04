from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from hparams import *
from nn.model import LSTMModel
from utils.utils import create_checkpoint_manager, load_checkpoint_if_exists
from utils.generate_helpers import generate_review

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

generated_review = generate_review(model, 'I really liked this tea', temperature=0.7)
print(generated_review)
