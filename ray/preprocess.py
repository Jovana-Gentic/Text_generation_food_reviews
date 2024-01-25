import pickle
from hparams import *
from data.preprocessor import vocabulary

vocabulary = vocabulary()
with open('vocabulary.pkl', 'wb') as fp:
    pickle.dump(vocabulary, fp)