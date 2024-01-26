import tensorflow as tf
from hparams import *
from nn.losses import MaskedSparseCategoricalCrossentropy
from nn.metrics import masked_sparse_categorical_accuracy
from data.data import get_word2idx, get_idx2word
from utils.utils import detokenize_x, detokenize_y, save_checkpoint, tensorboard_log


def get_tokens(logits):
    """
    Transforms logits to tokens with argmax and random sampling with gumbel noise trick.
    """
    y_pred_argmax = tf.argmax(logits, axis=-1, output_type=tf.int32)
    
    rand_uniform = tf.random.uniform(tf.shape(logits), minval=1e-5, maxval=1. - 1e-5)
    gumbel_noise = -tf.math.log(-tf.math.log(rand_uniform))
    y_pred_sampling = tf.argmax(logits + gumbel_noise, axis=-1, output_type=tf.int32)
    return y_pred_argmax, y_pred_sampling

def train_step_wrapper(model, loss_fn, optimizer):
    @tf.function(input_signature=[tf.TensorSpec(shape=[BATCH_SIZE, None], dtype=tf.int32), tf.TensorSpec(shape=[BATCH_SIZE, None], dtype=tf.int32)])
    def train_step( x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, logits
    return train_step

def val_step_wrapper(model, loss_fn):
    @tf.function(input_signature=[tf.TensorSpec(shape=[BATCH_SIZE, None], dtype=tf.int32), tf.TensorSpec(shape=[BATCH_SIZE, None], dtype=tf.int32)])
    def val_step(x,y):
        logits = model(x, training=False)
        loss_value = loss_fn(y, logits)
        return loss_value, logits
    return val_step

def validate(dataset_val, step, data_name, model, loss_fn, word2idx, idx2word):
    avg_val_loss = 0.
    avg_val_argmax_acc = 0.
    avg_val_sampling_acc = 0.

    val_step = val_step_wrapper(model, loss_fn)
    for i,(x_batch_val, y_batch_val) in zip(range(VALIDATION_STEPS),dataset_val):
        loss_value, logits = val_step(x_batch_val, y_batch_val) 
        avg_val_loss += loss_value 

        y_pred_argmax, y_pred_sampling = get_tokens(logits)
        val_argmax_acc = masked_sparse_categorical_accuracy(y_batch_val, y_pred_argmax, word2idx)
        val_sampling_acc = masked_sparse_categorical_accuracy(y_batch_val, y_pred_sampling, word2idx)
        avg_val_argmax_acc += val_argmax_acc
        avg_val_sampling_acc += val_sampling_acc

    avg_val_loss /= (i + 1)
    avg_val_argmax_acc /= (i + 1)
    avg_val_sampling_acc /= (i + 1)
    print(f"{data_name} loss at step {step}: {avg_val_loss:.4f}")
    print(f"{data_name} argmax acc at step {step}: {avg_val_argmax_acc:.2f}")
    print(f"{data_name} sampling acc at step {step}: {avg_val_sampling_acc:.2f}")

    limit = detokenize_x(y_batch_val[0].numpy(), word2idx, idx2word)
    detokenize_y(y_pred_argmax[0].numpy(), limit, 'Argmax: ', idx2word)
    detokenize_y(y_pred_sampling[0].numpy(), limit, 'Sampling: ', idx2word) 
    
    return avg_val_loss, avg_val_argmax_acc, avg_val_sampling_acc

def train(dataset_train, dataset_val, train_summary_writer, val_summary_writer, manager, model, optimizer):
    loss_fn = MaskedSparseCategoricalCrossentropy()
    word2idx = get_word2idx()
    idx2word = get_idx2word()
    initial_step = manager.checkpoint.checkpoint_starter.read_value().numpy()
    best_val_loss = 2**16
    pbar = tf.keras.utils.Progbar(EVAL_EVERY_N_STEPS, stateful_metrics=['loss'])
    
    print('Training started! \U0001F973 \n_____________________________________________________')
    train_step = train_step_wrapper(model, loss_fn, optimizer)
    for step, (x_batch_train, y_batch_train) in zip(range(initial_step+1,TOTAL_TRAINING_STEPS+1),dataset_train):
        loss_value, logits = train_step(x_batch_train, y_batch_train) 

        pbar.update(step % EVAL_EVERY_N_STEPS, [('loss', loss_value)])

        if step % EVAL_EVERY_N_STEPS == 0:
            print(f"\nSeen so far: {(step + 1) * BATCH_SIZE} samples")

            avg_train_loss, avg_train_argmax_acc, avg_train_sampling_acc = validate(dataset_train, step, 
                                                                                    data_name='Training', model=model, 
                                                                                    loss_fn=loss_fn,word2idx=word2idx, idx2word=idx2word)    
            avg_val_loss, avg_val_argmax_acc, avg_val_sampling_acc = validate(dataset_val, step, 
                                                                              data_name='Validation', model=model, 
                                                                              loss_fn=loss_fn,word2idx=word2idx, idx2word=idx2word)

            tensorboard_log(train_summary_writer, avg_train_loss, avg_train_argmax_acc, avg_train_sampling_acc, step)
            tensorboard_log(val_summary_writer, avg_val_loss, avg_val_argmax_acc, avg_val_sampling_acc, step)
            print(f'SAVED LOGS TO TENSORBOARD FOR STEP {step}..')

            if avg_val_loss <= best_val_loss:      
                best_val_loss = avg_val_loss
                save_checkpoint(step, manager)
                print(f'SAVED CHECKPOINT FOR STEP {step}..')

            print("_____________________________________________________")   

        if step == TOTAL_TRAINING_STEPS:
            print('Training complete! \U0001F929')
