"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-04-09

MODEL TRAINING

Implementation of two training function:
 - "Vanilla" seq2seq model
 - GAN seq2seq.
"""
import os
import time
import pickle
from pdb import set_trace as BP

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# local modules
import deterioration
import tools


def process_batch(batch, params):
    '''
    Process a mini batch for training, works for Train anv Validation data.
    Steps:
    Trim NaN's left (after processing pipeline, they should all be the right
    ones). Trends too short already filtered in processing.py.
    Apply artificial deterioration.
    Process to RNN format ('sliding window' to input series) and pack into final array.
    Fill NaN's 'with placeholder_value'.
    '''
    import numpy as np
    import deterioration, tools  # local modules

    batch = [ np.isfinite(batch[ i , : ]) for i in range(batch.shape[0]) ]
    batch = [ deterioration.apply(series, params) for series in batch ]
    batch = [ tools.RNN_univariate_processing(series, len_input = params['len_input']) for series in batch ]
    batch = np.concatenate(batch)
    batch[ np.isnan(batch) ] = params['placeholder_value']

    # Subsample mini batch to allow faster training and higher stochasticity of training
    batch = batch[ np.random.choice(batch.shape[0], params['batch_size'], replace = False) , : ]

    # ANN requires shape: ( n obs , len input , 1 )
    batch = np.expand_dims(batch, axis = -1)

    return batch


def train(model, X, V, params):
    '''
    Trains 'Vanilla' Seq2seq model.
    To facilitate training, each time series (dataset row) is loaded and processed to a 
    2D array for RNNs, then a batch of size params['batch_size'] is sampled randomly from it.
    This trick doesn't train the model on all dataset on a single epoch, but allows it to be
    fed with data from all Train set in reasonable amounts of time.
    Acutal training step is under a @tf.funcion decorator; this reduced the whole train 
    step to a tensorflow op, to speed up training.
    History vectors for Training and Validation loss are pickled to /saved_models/ folder.
    '''
    import time
    import numpy as np
    import tensorflow as tf

    X_index = np.array(range(X.shape[0]))  # index X for faster fetch batch and shuffle

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_on_batch(batch):
        with tf.GradientTape() as tape:
            current_loss = loss(batch, model(batch))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    ## Training session starts
    #train_loss_history = []
    #val_loss_history = []

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling row index
        if params['shuffle']:
            X_index = X_index[ np.random.choice(len(X_index), len(X_index), replace = False) ]

        for iteration in range(X.shape[0]):
            start = time.time()

            # fetch batch and train
            batch_index = X_index[ iteration:iteration+1 ]
            current_batch = X[ batch_index , : ]
            current_batch = process_batch(current_batch, params)

            current_loss = train_on_batch(current_batch)

            # Save and print progress each 50 training steps
            if iteration % 50 == 0:
                v_sample = np.random.choice(V.shape[0])
                V_batch = V[ v_sample:v_sample+1 , : ]
                V_batch = process_batch(V_batch, params)
                validation_loss = loss(V_batch, model(V_batch))

                #train_loss_history.append(current_loss)
                #val_loss_history.append(validation_loss)

                print('{}.{}   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
                    epoch, iteration, current_loss, validation_loss, round(time.time()-start, 4)))

    print('\nTraining complete.\n')

    model.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Model saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    return None



################################################################################################
###    GAN TRAINING IS STILL A WORK IN PROGRESS - DO NOT TOUCH UNTIL VANILLA TRAINING IS READY
################################################################################################

# def train_GAN(generator, discriminator, X, V, params):
#     import time
#     import numpy as np
#     from sklearn.utils import shuffle
#     import tensorflow as tf
#
#     generator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
#     discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
#
#     ## TRAINING FUNCTIONS
#     @tf.function
#     def train_generator(X_real, X_imputed, prediction_imputed, classification_weight, regression_weight):
#         '''
#         Args:
#         - X_real:                 true time series,
#         - X_imputed:              generator's prediction,
#         - prediction_imputed:     discriminator's evaluation of generator
#         - classification_weight:  weigth of generator's ability to fool discriminator in final Loss sum
#         - regression_weight:      weight of regression quality in final Loss sum
#         '''
#         with tf.GrandientTape() as generator_tape:
#             classification_loss = tf.keras.losses.BinaryCrossentropy(tf.ones_like(prediction_imputed), prediction_imputed)
#             regression_loss = tf.keras.losses.MeanAbsoluteError(X_real, X_imputed)
#
#             ## AGGIUNGI BLOCCO PER CONTROLLO
#             tf.print('\nCHECK: classification_loss and regression_loss:')
#             tf.print(classification_loss)
#             tf.print(regression_loss)
#
#             generator_current_loss = classification_loss * classification_weight + regression_loss * regression_weight
#         generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
#         generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
#         return generator_current_loss
#
#     @tf.function
#     def train_discriminator(prediction_real, prediction_imputed):
#         with tf.GrandientTape() as discriminator_tape:
#             loss_real = tf.keras.losses.BinaryCrossentropy(tf.ones_like(prediction_real), prediction_real)
#             loss_imputed = tf.keras.losses.BinaryCrossentropy(tf.zeros_like(prediction_imputed), prediction_imputed)
#             discriminator_current_loss = loss_real + loss_imputed
#         dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)
#         discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))
#         return discriminator_current_loss
#
#     ## TRAINING
#     for epoch in range(params['n_epochs']):
#         start = time.time()
#
#         # if params['shuffle']:
#         #     ### AGGIUNGI SHUFFLE VELOCE, CON INDICE
#
#         for iteration in range(X.shape[0]//batch_size):
#
#             # Take batch and apply artificial deterioration to impute data
#             take = iteration * params['batch_size']
#             X_real = X[ take:take+params['batch_size'] , : ]
#             X_imputed = deterioration.apply(X_real)
#             X_imputed = generator(X_imputed)
#
#             ## TRAIN DICRIMINATOR
#             generator.trainable = False
#             discriminator.trainable = True
#
#             # Generate Discriminator's predictions (needed for both G and D losses)
#             prediction_real = discriminator(X_real)
#             prediction_imputed = discriminator(X_imputed)
#
#             discriminator_current_loss = train_discriminator(prediction_real, prediction_imputed)
#
#             ## TRAIN GENERATOR
#             generator.trainable = False
#             discriminator.trainable = True
#
#             generator_current_loss = train_generator(X_real, X_imputed, prediction_imputed, classification_weight, regression_weight)
#
#         print('{} - {}.  \t Generator Loss: {}.  \t Discriminator Loss: {}.  \t  Time: {}ss'.format(
#             epoch, generator_current_loss, discriminator_current_loss, round(start - time.time(), 2)))
#
#     print('Training complete.\n')
#
#     model.save('{}/{}.h5'.format(params['save_path'], params['model_name']))
#     print('Model saved at:\n\t{}'.format(params['save_path']))
#     return None
