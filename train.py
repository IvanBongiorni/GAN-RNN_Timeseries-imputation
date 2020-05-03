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

# local modules
import deterioration
import tools


def fetch_processed_batch(X_train, X_index, iteration, batch_size):
    import numpy as np

    # local modules
    import deterioration
    import tools

    take = iteration * batch_size
    batch_index = X_index[ take:take + batch_size , : ]
    X_batch = [ X_train[ i , : ] for i in batch_index ]

    # Trim NaN's left (after processing pipeline, they should all be the right ones)
    X_batch = [ series[ np.isfinite(series) ] for series in  ]  # Trends not long enough have already been filtered in processing.py

    # Apply artificial deterioration
    X_batch = [ deterioration.apply(series, params) for series in X_batch ]

    # Process to RNN format ('sliding window' to input series) and pack into final array
    X_batch = [ tools.RNN_univariate_processing(series, len_input = params['len_input']) for series in array ]

    X_batch = np.concatenate(X_batch)

    # Fill NaN's 'with placeholder_value'
    X_batch[ np.isnan(X_batch) ] = params['placeholder_value']

    return X_batch


def train(model, params, X_train, X_val):
    import time
    import numpy as np
    import tensorflow as tf

    # local modules
    import deterioration
    import tools

    # I will use this index to speed up fetching mini batches and reshuffles
    X_index = np.array(range(X_train.shape[0]))

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_on_batch(X_batch):
        with tf.GrandientTape() as tape:
            current_loss = loss(X_batch, model(X_batch))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    for epoch in range(params['n_epochs']):
        start = time.time()

        if params['shuffle']:
            shuffle = np.random.choice(len(X_index), len(X_index), replace = False)
            X_index = X_index[ shuffle ]

        for iteration in range(X_train.shape[0] // params['batch_size']):
            X_batch = fetch_processed_batch(X_train, X_index, iteration, params['batch_size'])
            current_loss = train_on_batch(X_batch)

    # Sample some val_batch

    ### IMPORTANT: VALIDATION LOSS MUST BE EXECUTED
    X_val = fetch_validation_batch()
    validation_loss = loss(X_val, model(X_val))

    print('{}.   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
        epoch,
        current_loss.numpy(),
        validation_loss.numpy(),
        round(time.time()-start, 2)
    ))
    print('Training complete.\n')

    model.save('{}/{}.h5'.format(params['save_path'], params['model_name']))
    print('Model saved at:\n\t{}'.format(params['save_path']))
    return None


def train_GAN(generator, discriminator, X, V, params):
    import time
    import numpy as np
    from sklearn.utils import shuffle
    import tensorflow as tf

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    ## TRAINING FUNCTIONS
    @tf.function
    def train_generator(X_real, X_imputed, prediction_imputed, classification_weight, regression_weight):
        '''
        Args:
        - X_real:                 true time series,
        - X_imputed:              generator's prediction,
        - prediction_imputed:     discriminator's evaluation of generator
        - classification_weight:  weigth of generator's ability to fool discriminator in final Loss sum
        - regression_weight:      weight of regression quality in final Loss sum
        '''
        with tf.GrandientTape() as generator_tape:
            classification_loss = tf.keras.losses.BinaryCrossentropy(tf.ones_like(prediction_imputed), prediction_imputed)
            regression_loss = tf.keras.losses.MeanAbsoluteError(X_real, X_imputed)

            ## AGGIUNGI BLOCCO PER CONTROLLO
            tf.print('\nCHECK: classification_loss and regression_loss:')
            tf.print(classification_loss)
            tf.print(regression_loss)

            generator_current_loss = classification_loss * classification_weight + regression_loss * regression_weight
        generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        return generator_current_loss

    @tf.function
    def train_discriminator(prediction_real, prediction_imputed):
        with tf.GrandientTape() as discriminator_tape:
            loss_real = tf.keras.losses.BinaryCrossentropy(tf.ones_like(prediction_real), prediction_real)
            loss_imputed = tf.keras.losses.BinaryCrossentropy(tf.zeros_like(prediction_imputed), prediction_imputed)
            discriminator_current_loss = loss_real + loss_imputed
        dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))
        return discriminator_current_loss

    ## TRAINING
    for epoch in range(params['n_epochs']):
        start = time.time()

        if params['shuffle']:
            ### AGGIUNGI SHUFFLE VELOCE, CON INDICE

        for iteration in range(X.shape[0]//batch_size):

            # Take batch and apply artificial deterioration to impute data
            take = iteration * params['batch_size']
            X_real = X[ take:take+params['batch_size'] , : ]
            X_imputed = deterioration.apply(X_real)
            X_imputed = generator(X_imputed)

            ## TRAIN DICRIMINATOR
            generator.trainable = False
            discriminator.trainable = True

            # Generate Discriminator's predictions (needed for both G and D losses)
            prediction_real = discriminator(X_real)
            prediction_imputed = discriminator(X_imputed)

            discriminator_current_loss = train_discriminator(prediction_real, prediction_imputed)

            ## TRAIN GENERATOR
            generator.trainable = False
            discriminator.trainable = True

            generator_current_loss = train_generator(X_real, X_imputed, prediction_imputed, classification_weight, regression_weight)

        print('{} - {}.  \t Generator Loss: {}.  \t Discriminator Loss: {}.  \t  Time: {}ss'.format(
            epoch, generator_current_loss, discriminator_current_loss, round(start - time.time(), 2)))

    print('Training complete.\n')

    model.save('{}/{}.h5'.format(params['save_path'], params['model_name']))
    print('Model saved at:\n\t{}'.format(params['save_path']))
    return None
