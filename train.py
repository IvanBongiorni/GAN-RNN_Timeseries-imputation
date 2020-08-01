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
    return X_batch, Y_batch, mask


def train_vanilla_seq2seq(model, params):
    '''
    ## TODO:  [ doc to be rewritten ]
    '''
    import time
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def train_on_batch(X_batch, Y_batch, mask):
        mask = tf.expand_dims(mask, axis=-1)

        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(tf.math.abs(
                tf.math.multiply(model(X_batch), mask) - tf.math.multiply(Y_batch, mask)))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

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

                mask = np.expand_dims(mask, axis=-1)
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
    ## TODO:  [ doc to be rewritten ]
    '''
    import time
    import numpy as np
    import tensorflow as tf

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # this works for both G and D
    # MAE = tf.keras.losses.MeanAbsoluteError()  # to check Validation performance

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    @tf.function
    def generator_loss(discriminator_guess_fakes):
        return cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)

    # Label smoothing
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

            # that's complex: it prepares an imputed batch for the Discriminator. It removes the first
            # variable (at [:,:,0]) that is the deteriorated trend, and puts the imputation made by the Generator
            generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)

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
            X_batch, Y_batch, mask = process_series(batch, params)

            # Load another series of real observations ( this block is a subset of process_series() )
            real_example = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(),  np.random.choice(np.delete(X_files, iteration))))
            real_example = tools.RNN_multivariate_processing(array=real_example, len_input=params['len_input'])
            sample = np.random.choice(real_example.shape[0], size=np.min([real_example.shape[0], params['batch_size']]), replace=False)
            real_example = real_example[sample,:,:]
            # real_example = np.expand_dims(real_example, axis=-1)

            generator_current_loss, discriminator_current_loss = train_step(X_batch, real_example)

            if iteration % 100 == 0:

                # Check Imputer's plain Loss on training example
                generator_imputation = generator(X_batch)
                train_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(tf.squeeze(generator_imputation), mask) - tf.math.multiply(tf.squeeze(Y_batch), mask)
                ))

                # get Generative and Aversarial Losses (and binary accuracy)
                generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                discriminator_guess_fakes = discriminator(generator_imputation)
                discriminator_guess_reals = discriminator(real_example)

                # Add imputation Loss on Validation data
                v_file = np.random.choice(V_files)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                X_batch, Y_batch, mask = process_series(batch, params)
                generator_imputation = generator(X_batch)
                # generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                val_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(tf.squeeze(generator_imputation), mask) - tf.math.multiply(tf.squeeze(Y_batch), mask)
                ))

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


def train_partial_GAN(generator, discriminator, params):
    '''
    ## TODO:  [ doc to be rewritten ]
    '''
    import time
    import numpy as np
    import tensorflow as tf

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True) # this works for both G and D
    MAE = tf.keras.losses.MeanAbsoluteError()  # to check Validation performance

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def train_step(X_batch, Y_batch, real_example, mask, w):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            generator_imputation = generator(X_batch)
            # that's complex: it prepares an imputed batch for the Discriminator. It removes the first
            # variable (at [:,:,0]) that is the deteriorated trend, and puts the imputation made by the Generator
            generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)

            discriminator_guess_fakes = discriminator(generator_imputation)
            discriminator_guess_reals = discriminator(real_example)

            # Generator loss
            mask = tf.expand_dims(mask, axis=-1)
            g_loss_mae = tf.reduce_mean(tf.math.abs(
                tf.math.multiply(generator(X_batch), mask) - tf.math.multiply(Y_batch, mask)))
            g_loss_gan = cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)

            # tf.print('mae:', g_loss_mae, '; gan:', g_loss_gan)

            generator_current_loss = g_loss_mae + (g_loss_gan * w)  # magnitude of GAN loss to be adjusted

            # Disccriminator loss - Label smoothing
            # d_loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
            # d_loss_reals = cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)
            loss_fakes = cross_entropy(
                tf.random.uniform(shape=tf.shape(discriminator_guess_fakes), minval=0.0, maxval=0.2), discriminator_guess_fakes
            )
            loss_reals = cross_entropy(
                tf.random.uniform(shape=tf.shape(discriminator_guess_reals), minval=0.8, maxval=1), discriminator_guess_reals
            )
            discriminator_current_loss = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes) + cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)

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
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace = False) ]

        for iteration in range(X_files.shape[0]):
        # for iteration in range( int(X_files.shape[0] * 0.1) ):      ### TEMPORARY TEST
            start = time.time()

            # fetch batch by filenames index and train
            batch = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), X_files[iteration]) )
            X_batch, Y_batch, mask = process_series(batch, params)

            # Load another series of real observations ( this block is a subset of process_series() )
            real_example = np.load( '{}/data_processed/Training/{}'.format(os.getcwd(), np.random.choice(np.delete(X_files, iteration))))
            real_example = tools.RNN_multivariate_processing(real_example, len_input = params['len_input'])
            sample = np.random.choice(real_example.shape[0], size = np.min([real_example.shape[0], params['batch_size']]), replace = False)
            real_example = real_example[ sample , : ]

            ### TODO: UNIFORMARE QUESTI INPUT CON LA FUNZIONE SOPRA
            generator_current_loss, discriminator_current_loss = train_step(X_batch, Y_batch, real_example, mask, params['loss_weight'])

            if iteration % 100 == 0:
                # To get Generative and Aversarial Losses (and binary accuracy)
                # for Generator simply repeat what's in train_step() above
                generator_imputation = generator(X_batch)
                generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                discriminator_guess_reals = discriminator(real_example)
                discriminator_guess_fakes = discriminator(generator_imputation)

                # Check Imputer's plain Loss on training example
                # train_loss = MAE(batch, generator(deteriorated))
                train_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(generator(X_batch), tf.expand_dims(mask, axis=-1)) - tf.math.multiply(Y_batch, tf.expand_dims(mask, axis=-1))))

                # Add imputation Loss on Validation data
                v_file = np.random.choice(V_files)
                batch = np.load( '{}/data_processed/Validation/{}'.format(os.getcwd(), v_file) )
                X_batch, Y_batch, mask = process_series(batch, params)
                val_loss = tf.reduce_mean(tf.math.abs(
                    tf.math.multiply(generator(X_batch), tf.expand_dims(mask, axis=-1)) - tf.math.multiply(Y_batch, tf.expand_dims(mask, axis=-1))))

                print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy (reals, fakes): ({}, {})   \tTime: {}ss'.format(
                    epoch, iteration,
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)),
                    round(time.time()-start, 4)
                ))
                print('\t\tImputation Loss: {}   \tValidation Loss: {}\n'.format(train_loss, val_loss))

    print('\nTraining complete.\n')

    generator.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Generator saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    if params['save_discriminator']:
        discriminator.save('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name']))
        print('\nDiscriminator saved at:\n{}'.format('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name'])))

    return None
