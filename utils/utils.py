import tensorflow as tf
from nltk.tokenize.treebank import TreebankWordDetokenizer
from hparams import *

def create_checkpoint_manager(checkpoint_path, model, optimizer, max_allowed_checkpoints=5):
    checkpoint_starter = tf.Variable(0, trainable=False, dtype=tf.int32)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, checkpoint_starter=checkpoint_starter)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_path, max_to_keep=max_allowed_checkpoints)    
    return manager

def save_checkpoint(step, checkpoint_manager):
    checkpoint_manager.checkpoint.checkpoint_starter.assign(step)
    checkpoint_manager.save(step)
    
def load_checkpoint_if_exists(manager):
    checkpoint_path = manager.directory
    checkpoint = manager.checkpoint
    try:
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            print('Loading checkpoint from: {}'.format(checkpoint_state.model_checkpoint_path))
            checkpoint.restore(checkpoint_state.model_checkpoint_path)
        
        else:
            print('No model to load at {}'.format(checkpoint_path))
    
    except tf.errors.OutOfRangeError as e:
        raise tf.errors.OutOfRangeError('Cannot restore checkpoint: {}'.format(e))
    
def tensorboard_log(summary_writer, loss_value, argmax_accuracy, sampling_accuracy, step):
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step)
        tf.summary.scalar('argmax_accuracy', argmax_accuracy, step=step)
        tf.summary.scalar('sampling_accuracy', sampling_accuracy, step=step)
    summary_writer.flush()

def detokenize_x(x, word2idx, idx2word):
    """
    Turns tokens into words for our input samples.
    """
    token_list_x = []
    for token in x:
        if token != word2idx['<PAD>']:
            word = idx2word[token]
            token_list_x.append(word)
    limit = len(token_list_x)
    print('Target: ',TreebankWordDetokenizer().detokenize(token_list_x))
    print('')
    return limit

def detokenize_y(y, limit, argmax_or_sampling, idx2word):
    """
    Turns tokens into words of length of input sample for output samples.
    """
    token_list_y = []
    for token in y[:limit]:
        word = idx2word[token]
        token_list_y.append(word)
    print(argmax_or_sampling,TreebankWordDetokenizer().detokenize(token_list_y))
    print('')    