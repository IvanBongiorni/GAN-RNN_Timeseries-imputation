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
    sample = np.random.choice(series.shape[0], size = np.min([series.shape[0], params['batch_size']]), replace = False)
    series = series[ sample , : ]
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


def train_GAN(generator, discriminator, params):
    '''
    This function trains a pure Generative Adversarial Network.
    The main differences between a canonical GAN (as formulated by Goodfellow [2014]) and the one
    implemented here is that the generation (imputation) produced doesn't stem from pure Gaussian
    noise, but from artificially deteriorated trends. A randomic and an 'epistemic' component
    coexist therefore.
    Discriminator's accuracy metrics at the bottom is expressed as the only fake-detecting accuracy.
    '''
    import time
    import numpy as np
    import tensorflow as tf

    cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits = True) # this works for both G and D

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def generator_loss(discriminator_guess_fakes):
        return cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)

    @tf.function
    def discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes, real_example):
        loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
        loss_real = cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)
        return loss_fakes + loss_real

    @tf.function
    def train_step(deteriorated, real_example):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            generator_imputation = generator(deteriorated)

            discriminator_guess_fakes = discriminator(generator_imputation)
            discriminator_guess_reals = discriminator(real_example)

            generator_current_loss = generator_loss(discriminator_guess_fakes)
            discriminator_current_loss = discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes, real_example)

        generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
        dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))

        return generator_current_loss, discriminator_current_loss


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

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            batch, deteriorated = process_series(batch, params)

            # Load another series of real observations ( this block is a subset of process_series() )
            real_example = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(),  np.random.choice(np.delete(X_files, iteration))))
            real_example = real_example[ np.isfinite(real_example) ] # only right-trim NaN's. Others were removed in processing
            real_example = tools.RNN_univariate_processing(real_example, len_input = params['len_input'])
            sample = np.random.choice(real_example.shape[0], size = np.min([real_example.shape[0], params['batch_size']]), replace = False)
            real_example = real_example[ sample , : ]
            real_example = np.expand_dims(real_example, axis = -1)

            generator_current_loss, discriminator_current_loss = train_step(deteriorated, real_example)

            # Report progress
            if iteration % 50 == 0:
                # To get Discriminator's binary accuracy
                generator_imputation = generator(deteriorated)
                discriminator_guess_fakes = discriminator(generator_imputation)

                print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy: {}   \tTime: {}ss'.format(
                    epoch, iteration,
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)),
                    round(time.time()-start, 4)
                ))

    print('\nTraining complete.\n')

    generator.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Generator saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    if params['save_discriminator']:
        discriminator.save('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name']))
        print('\nDiscriminator saved at:\n{}'.format('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name'])))

    return None



#######################################################################################################
###    PARTIALLY ADVERSARIAL NETWORK is still a WORK IN PROGRESS - DO NOT USE THIS CODE UNTIL READY
#######################################################################################################



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
