"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-04-09

MODEL TRAINING

Implementation of three training functions:
 - "Vanilla" seq2seq model
 - GAN seq2seq.
 - Partially adversarial seq2seq
"""
import os
import time
from pdb import set_trace as BP

import numpy as np
import tensorflow as tf

# local modules
import deterioration
import tools


def process_series(series, params):
    import numpy as np
    import deterioration, tools  # local imports

    series = series[ np.isfinite(series) ] # only right-trim NaN's. Others were removed in processing
    series = tools.RNN_univariate_processing(series, len_input = params['len_input'])

    try:
        sample = np.random.choice(series.shape[0], params['batch_size'], replace = False)
        series = series[ sample , : ]
    except:
        pass

    deteriorated = np.copy(series)
    deteriorated = deterioration.apply(deteriorated, params)
    deteriorated[ np.isnan(deteriorated) ] = params['placeholder_value']

    # ANN requires shape: ( n obs , len input , 1 )
    series = np.expand_dims(series, axis = -1)
    deteriorated = np.expand_dims(deteriorated, axis = -1)

    return series, deteriorated


def train_vanilla_seq2seq(model, params):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_on_batch(batch, deteriorated):
        with tf.GradientTape() as tape:
            current_loss = loss(batch, model(deteriorated))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    ## Training session starts

    #train_loss_history = []
    #val_loss_history = []

    for epoch in range(params['n_epochs']):

        # Get list of all Training and Validation observations
        X_files = os.listdir( os.getcwd() + '/data_processed/Training/' )
        if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
        if '.gitignore' in X_files: X_files.remove('.gitignore')
        X_files = np.array(X_files)

        V_files = os.listdir( os.getcwd() + '/data_processed/Validation/' )
        if 'readme_validation.md' in V_files: V_files.remove('readme_validation.md')
        if '.gitignore' in V_files: V_files.remove('.gitignore')
        V_files = np.array(V_files)

        # Sample subset from it if requested, otherwise use all data
        #if params['train_size_per_epoch'] is not None:
        #    X_files = X_files[ np.random.choice(X_files.shape[0], size = params['train_size_per_epoch'], replace = False) ]

        # Shuffle data by shuffling row index
        # if params['shuffle']:
        #     X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace = False) ]

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            batch, deteriorated = process_series(batch, params)

            current_loss = train_on_batch(batch, deteriorated)

            # Save and print progress each 50 training steps
            if iteration % 50 == 0:

                v_file = os.listdir( os.getcwd() + '/data_processed/Validation/' )
                if 'readme_validation.md' in v_file: v_file.remove('readme_validation.md')
                if '.gitignore' in v_file: v_file.remove('.gitignore')
                v_file = np.random.choice(v_file)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                batch, deteriorated = process_series(batch, params)

                validation_loss = loss(batch, model(deteriorated))

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

# def train_GAN():
#     return None


# def train_partial_GAN(generator, discriminator, X, V, params):
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
