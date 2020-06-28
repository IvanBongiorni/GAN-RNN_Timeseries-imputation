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


def process_series(batch, params):
    import numpy as np
    import deterioration, tools  # local imports

    X_batch = tools.RNN_multivariate_processing(array=batch, len_input=params['len_input'])
    sample = np.random.choice(X_batch.shape[0], size=np.min([X_batch.shape[0], params['batch_size']]), replace = False)
    X_batch = X_batch[ sample,:,: ]
    Y_batch = np.copy(X_batch[:,:,0])
    # X_batch[:,:,0] = deterioration.apply(X_batch[:,:,0], params)
    # X_batch[ np.isnan(X_batch) ] = params['placeholder_value']
    mask = deterioration.mask(X_batch[:,:,0], params)
    X_batch[:,:,0] = np.where(mask==1, params['placeholder_value'], X_batch[:,:,0])

    Y_batch = np.expand_dims(Y_batch, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return X_batch, Y_batch, mask


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
    import tensorflow.keras.backend as K

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    # loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_on_batch(X_batch, Y_batch, mask):
        with tf.GradientTape() as tape:
            # current_loss = loss(Y_batch, model(X_batch))
            current_loss = tf.reduce_mean(tf.math.abs(
                tf.math.multiply(model(X_batch), mask) - tf.math.multiply(Y_batch, mask)))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    ## Training session starts

    # Get list of all Training and Validation observations
    X_files = os.listdir( os.getcwd() + '/data_processed/Training/' )
    if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
    if '.gitignore' in X_files: X_files.remove('.gitignore')
    X_files = np.array(X_files)

    V_files = os.listdir( os.getcwd() + '/data_processed/Validation/' )
    if 'readme_validation.md' in V_files: V_files.remove('readme_validation.md')
    if '.gitignore' in V_files: V_files.remove('.gitignore')
    V_files = np.array(V_files)

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling row index
        if params['shuffle']:
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace=False) ]

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            X_batch, Y_batch, mask = process_series(batch, params)
            
            current_loss = train_on_batch(X_batch, Y_batch, mask)

            # Save and print progress each 50 training steps
            if iteration % 100 == 0:
                v_file = np.random.choice(V_files)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                X_batch, Y_batch, mask = process_series(batch, params)

                validation_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(model(X_batch), mask) - tf.math.multiply(Y_batch, mask)))

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

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # this works for both G and D
    MAE = tf.keras.losses.MeanAbsoluteError()  # to check Validation performance

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    @tf.function
    def generator_loss(discriminator_guess_fakes):
        return cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)

    # @tf.function
    # def discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes):
    #     loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
    #     loss_real = cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)
    #     return loss_fakes + loss_real

    @tf.function
    def discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes):
        loss_fakes = cross_entropy(
            tf.random.uniform(shape=tf.shape(discriminator_guess_fakes), minval=0.0, maxval=0.2), discriminator_guess_fakes
        )
        # loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
        loss_reals = cross_entropy(
            tf.random.uniform(shape=tf.shape(discriminator_guess_reals), minval=0.8, maxval=1), discriminator_guess_reals
        )
        return loss_fakes + loss_reals

    @tf.function
    def train_step(X_batch, real_example):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            generator_imputation = generator(X_batch)

            discriminator_guess_fakes = discriminator(generator_imputation)
            discriminator_guess_reals = discriminator(real_example)

            generator_current_loss = generator_loss(discriminator_guess_fakes)
            discriminator_current_loss = discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes)

        generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
        dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))

        return generator_current_loss, discriminator_current_loss

    # Get list of all Training and Validation observations
    X_files = os.listdir( os.getcwd() + '/data_processed/Training/' )
    if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
    if '.gitignore' in X_files: X_files.remove('.gitignore')
    X_files = np.array(X_files)

    V_files = os.listdir( os.getcwd() + '/data_processed/Validation/' )
    if 'readme_validation.md' in V_files: V_files.remove('readme_validation.md')
    if '.gitignore' in V_files: V_files.remove('.gitignore')
    V_files = np.array(V_files)

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling row index
        if params['shuffle']:
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace=False) ]

        for iteration in range(X_files.shape[0]):
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            X_batch, Y_batch = process_series(batch, params)

            # Load another series of real observations ( this block is a subset of process_series() )
            real_example = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(),  np.random.choice(np.delete(X_files, iteration))))
            real_example = tools.RNN_multivariate_processing(array=real_example, len_input=params['len_input'])
            sample = np.random.choice(real_example.shape[0], size=np.min([real_example.shape[0], params['batch_size']]), replace=False)
            real_example = real_example[sample,:,:]
            real_example = np.expand_dims(real_example, axis=-1)

            generator_current_loss, discriminator_current_loss = train_step(X_batch, real_example)

            if iteration % 100 == 0:
                # To get Generative and Aversarial Losses (and binary accuracy)
                generator_imputation = generator(X_batch)
                discriminator_guess_reals = discriminator(real_example)
                discriminator_guess_fakes = discriminator(generator_imputation)

                # Check Imputer's plain Loss on training example
                train_loss = MAE(batch, generator(deteriorated))

                # Add imputation Loss on Validation data
                v_file = np.random.choice(V_files)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                X_batch, Y_batch = process_series(batch, params)
                val_loss = MAE(Y_batch, generator(X_batch))

                print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy (reals, fakes): ({}, {})   \tTime: {}ss'.format(
                    epoch, iteration,
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)),
                    round(time.time()-start, 4)
                ))
                print('\t\tTraining Loss: {}   \tValidation Loss: {}\n'.format(train_loss, val_loss))

    print('\nTraining complete.\n')

    generator.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Generator saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    if params['save_discriminator']:
        discriminator.save('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name']))
        print('\nDiscriminator saved at:\n{}'.format('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name'])))

    return None


# def train_partial_GAN(generator, discriminator, params):
#     '''
#     Modification of train_GAN(), with the addition of MAE Loss for Generator.
#     Now Generator's Loss is composed of two elements: a canonical regression Loss
#     (MAE), and he GAN Loss obtained from trying to fool the Discriminator. Since
#     their magnitude are quite different, a balance in the final loss values must be
#     achieved through artificial weighting, determined by params['loss_weight'].
#     '''
#     import time
#     import numpy as np
#     import tensorflow as tf
#
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True) # this works for both G and D
#     MAE = tf.keras.losses.MeanAbsoluteError()  # to check Validation performance
#
#     generator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
#     discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
#
#     @tf.function
#     def generator_loss(batch, deteriorated, discriminator_guess_fakes, w):
#         mae_loss = MAE(batch, deteriorated)
#         gan_loss = cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)
#
#         generator_loss = mae_loss * w + gan_loss * (1-w)
#         return generator_loss
#
#     @tf.function
#     def discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes):
#         loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
#         loss_reals = cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)
#
#         tf.print('Discriminator Loss, fakes, reals:', loss_fakes, loss_reals)
#
#         return loss_fakes + loss_reals
#
#     @tf.function
#     def train_step(batch, deteriorated, real_example, w):
#         with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
#
#             generator_imputation = generator(deteriorated)
#
#             discriminator_guess_fakes = discriminator(generator_imputation)
#             discriminator_guess_reals = discriminator(real_example)
#
#             generator_current_loss = generator_loss(batch, deteriorated, discriminator_guess_fakes, w)
#             discriminator_current_loss = discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes)
#
#         generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
#         dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)
#
#         generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
#         discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))
#
#         return generator_current_loss, discriminator_current_loss
#
#     # Get list of all Training and Validation observations
#     X_files = os.listdir( os.getcwd() + '/data_processed/Training/' )
#     if 'readme_training.md' in X_files: X_files.remove('readme_training.md')
#     if '.gitignore' in X_files: X_files.remove('.gitignore')
#     X_files = np.array(X_files)
#
#     V_files = os.listdir( os.getcwd() + '/data_processed/Validation/' )
#     if 'readme_validation.md' in V_files: V_files.remove('readme_validation.md')
#     if '.gitignore' in V_files: V_files.remove('.gitignore')
#     V_files = np.array(V_files)
#
#     for epoch in range(params['n_epochs']):
#
#         # Shuffle data by shuffling row index
#         if params['shuffle']:
#             X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace = False) ]
#
#         for iteration in range(X_files.shape[0]):
#         # for iteration in range( int(X_files.shape[0] * 0.1) ):      ### TEMPORARY TEST
#             start = time.time()
#
#             # fetch batch by filenames index and train
#             batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
#             batch, deteriorated = process_series(batch, params)
#
#             # Load another series of real observations ( this block is a subset of process_series() )
#             real_example = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), np.random.choice(np.delete(X_files, iteration))))
#             real_example = real_example[ np.isfinite(real_example) ] # only right-trim NaN's. Others were removed in processing
#             real_example = tools.RNN_univariate_processing(real_example, len_input = params['len_input'])
#             sample = np.random.choice(real_example.shape[0], size = np.min([real_example.shape[0], params['batch_size']]), replace = False)
#             real_example = real_example[ sample , : ]
#             real_example = np.expand_dims(real_example, axis = -1)
#
#             generator_current_loss, discriminator_current_loss = train_step(batch, deteriorated, real_example, params['loss_weight'])
#
#             if iteration % 100 == 0:
#                 # To get Generative and Aversarial Losses (and binary accuracy)
#                 generator_imputation = generator(deteriorated)
#                 discriminator_guess_reals = discriminator(real_example)
#                 discriminator_guess_fakes = discriminator(generator_imputation)
#
#                 # Check Imputer's plain Loss on training example
#                 train_loss = MAE(batch, generator(deteriorated))
#
#                 # Add imputation Loss on Validation data
#                 v_file = np.random.choice(V_files)
#                 batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
#                 batch, deteriorated = process_series(batch, params)
#                 val_loss = MAE(batch, generator(deteriorated))
#
#                 print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy (reals, fakes): ({}, {})   \tTime: {}ss'.format(
#                     epoch, iteration,
#                     generator_current_loss,
#                     discriminator_current_loss,
#                     tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
#                     tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)),
#                     round(time.time()-start, 4)
#                 ))
#                 print('\t\tImputation Loss: {}   \tValidation Loss: {}\n'.format(train_loss, val_loss))
#
#     print('\nTraining complete.\n')
#
#     generator.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
#     print('Generator saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))
#
#     if params['save_discriminator']:
#         discriminator.save('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name']))
#         print('\nDiscriminator saved at:\n{}'.format('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name'])))
#
#     return None
