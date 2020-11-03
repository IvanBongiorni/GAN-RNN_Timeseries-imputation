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


def process_series(x, params):
    import numpy as np
    import deterioration, tools  # local imports

    x = tools.RNN_multivariate_processing(array=x, len_input=params['len_input'])

    # For each trend, sample 1 obs. and fix shape
    x = x[ np.random.choice(x.shape[0]) , : , : ]
    x = np.expand_dims(x, axis=0)

    y = np.copy(x[:,:,0])
    y = np.expand_dims(y, axis=-1)

    m = deterioration.mask(x[:,:,0], params)
    x[:,:,0] = np.where(m==1, params['placeholder_value'], x[:,:,0])

    # Returning a list to allow for list comprehension in train()
    return [x, y, m]


def train_vanilla_seq2seq(model, params):
    '''
    Trains basic Seq2seq model. First, it lists all Training and Validation files,
    to be used for mini-batch process and creation.
    When it starts training iterations: loads a mini-batch, processes it for RNN
    input, and call train_on_batch() function. It's an Autograph function, its @tf.function
    decorator turns it into a TensorFlow op for faster run.
    Each n iterations repeat mini-batch loading and processing on Validation data, to
    check for progress in Loss.
    Finally saves model in './saved_models' directory.
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

    # 1. Get list of all Training and Validation observations
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

        for iteration in range(X_files.shape[0] // params['batch_size']):
            start = time.time()

            # 2. Fetch batch by filenames index and train
            start = iteration * params['batch_size']
            batch = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in X_files[start:start+params['batch_size']] ]
            batch = [ process_series(array, params) for array in batch ]

            # Extract X, Y and Mask and stack them in final arrays
            X_batch = [array[0] for array in batch]
            Y_batch = [array[1] for array in batch]
            mask = [array[2] for array in batch]
            X_batch = np.concatenate(X_batch)
            Y_batch = np.concatenate(Y_batch)
            mask = np.concatenate(mask)

            # Train model
            current_loss = train_on_batch(X_batch, Y_batch, mask)

            # Repeat processing on Validation data and print progress
            if iteration % 50 == 0:
                batch = np.random.choice(V_files, size=params['validation_batch_size'], replace=False)
                batch = [ np.load('{}/data_processed/Validation/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]
                batch = [ process_series(array, params) for array in batch ]

                # Extract X, Y and Mask and stack them in final arrays
                X_batch = [array[0] for array in batch]
                Y_batch = [array[1] for array in batch]
                mask = [array[2] for array in batch]
                X_batch = np.concatenate(X_batch)
                Y_batch = np.concatenate(Y_batch)
                mask = np.concatenate(mask)

                mask = np.expand_dims(mask, axis=-1)
                validation_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(model(X_batch), mask) - tf.math.multiply(Y_batch, mask)))

                print('{}.{}   \tTraining Loss: {}   \tValidation Loss: {}'.format(
                    epoch, iteration, current_loss, validation_loss))

    print('\nTraining complete.\n')

    model.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Model saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    return None


def train_GAN(generator, discriminator, params):
    '''
    Trains a standard GAN model. First, it lists all Training and Validation files,
    to be used for mini-batch process and creation.
    When it starts training iterations: loads two mini-batches, one for actual imputation,
    the other from as real obs. for Discriminator (sampled randomly from all available
    training obs. but the ones used in X_batch).
    Here, a train_step() function is called, where both Generator and Discriminator
    networks are trained as in classical GANs, with the addition of label smoothing as
    a regularization technique.
    It's an Autograph function, its @tf.function decorator turns it into a TensorFlow op
    for faster run.
    Each n iterations repeat mini-batch loading and processing on Validation data, to
    check for progress in Loss.
    Finally saves model in './saved_models' directory. As a default option from config.yaml,
    the Discriminator is saved as well as '{model_name}_discriminator' in the same directory.
    '''
    import time
    import numpy as np
    import tensorflow as tf

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # this works for both G and D

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
            X_files = X_files[ np.random.choice(len(X_files), len(X_files), replace=False) ]

        for iteration in range(X_files.shape[0] // params['batch_size']):

            # First, sample just filenames for mini-batches
            start = iteration * params['batch_size']
            batch = X_files[start:start+params['batch_size']]
            # then sample some real examples
            real_example = np.array(list(set(X_files)-set(batch)))
            real_example = real_example[ np.random.choice(len(V_files), size=params['batch_size'], replace=False) ]

            # Process raw data, extract X, Y and Mask and stack them in final arrays
            batch = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]
            batch = [ process_series(array, params) for array in batch ]
            X_batch = [array[0] for array in batch]
            Y_batch = [array[1] for array in batch]
            mask = [array[2] for array in batch]
            X_batch = np.concatenate(X_batch)
            Y_batch = np.concatenate(Y_batch)
            mask = np.concatenate(mask)

            # For reals, you don't need x-y split and mask, therefore replicate only part of process_series()
            real_example = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in real_example ]
            real_example = [ tools.RNN_multivariate_processing(array, len_input=params['len_input']) for array in real_example ]
            real_example = [ array[ np.random.choice(array.shape[0]) , : , : ] for array in real_example ]
            real_example = [ np.expand_dims(array, axis=0) for array in real_example ]
            real_example = np.concatenate(real_example)

            generator_current_loss, discriminator_current_loss = train_step(X_batch, real_example)

            # Repeat processing on Validation data and print progress
            if iteration % 50 == 0:

                # Check Imputer's plain Loss on training example
                generator_imputation = generator(X_batch)
                train_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(tf.squeeze(generator_imputation), mask) - tf.math.multiply(tf.squeeze(Y_batch), mask)))

                # get Generative and Aversarial Losses (and binary accuracy)
                generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                discriminator_guess_fakes = discriminator(generator_imputation)
                discriminator_guess_reals = discriminator(real_example)

                # Extract X, Y and Mask and stack them in final arrays
                batch = np.random.choice(V_files, size=params['validation_batch_size'], replace=False)
                batch = [ np.load('{}/data_processed/Validation/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]
                batch = [ process_series(array, params) for array in batch ]

                X_batch = [array[0] for array in batch]
                Y_batch = [array[1] for array in batch]
                mask = [array[2] for array in batch]
                X_batch = np.concatenate(X_batch)
                Y_batch = np.concatenate(Y_batch)
                mask = np.concatenate(mask)

                generator_imputation = generator(X_batch)
                val_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(tf.squeeze(generator_imputation), mask) - tf.math.multiply(tf.squeeze(Y_batch), mask)))

                print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy (reals, fakes): ({}, {})'.format(
                    epoch, iteration,
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes))
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
    Trains a more advanced GAN model. First, it lists all Training and Validation files,
    to be used for mini-batch process and creation.
    When it starts training iterations: loads two mini-batches, one for actual imputation,
    the other from as real obs. for Discriminator (sampled randomly from all available
    training obs. but the ones used in X_batch).
    Here, a train_step() function is called, where both Generator and Discriminator
    networks are trained. The Discriminator is trained as for classical GANs, while the Generator
    is trained with a composite loss: a regression loss as in 'Vanilla Seq2seq', and the
    classical GAN loss as in train_GAN(). Since the magnitude of the two losses is very
    different (with GAN loss being significantly greater), a scaling factor is balancing
    the two components. Label smoothing is used to compute the Discriminator's loss as
    aregularization technique.
    train_step() is an Autograph function, its @tf.function decorator turns it into a
    TensorFlow op for faster run.
    Each n iterations repeat mini-batch loading and processing on Validation data, to
    check for progress in Loss.
    Finally saves model in './saved_models' directory. As a default option from config.yaml,
    the Discriminator is saved as well as '{model_name}_discriminator' in the same directory.
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

            generator_current_loss = g_loss_mae + (g_loss_gan * w)  # magnitude of GAN loss to be adjusted

            # Disccriminator loss - Label smoothing
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
            X_files = X_files[ np.random.choice(len(X_files), len(X_files), replace=False) ]

        for iteration in range(X_files.shape[0] // params['batch_size']):

            # First, sample just filenames for mini-batches
            start = iteration * params['batch_size']
            batch = X_files[start:start+params['batch_size']]
            # then sample some real examples
            real_example = np.array(list(set(X_files)-set(batch)))
            real_example = real_example[ np.random.choice(len(V_files), size=params['batch_size'], replace=False) ]

            # Process raw data, extract X, Y and Mask and stack them in final arrays
            batch = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]
            batch = [ process_series(array, params) for array in batch ]
            X_batch = [array[0] for array in batch]
            Y_batch = [array[1] for array in batch]
            mask = [array[2] for array in batch]
            X_batch = np.concatenate(X_batch)
            Y_batch = np.concatenate(Y_batch)
            mask = np.concatenate(mask)

            # For reals, you don't need x-y split and mask, therefore replicate only part of process_series()
            real_example = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in real_example ]
            real_example = [ tools.RNN_multivariate_processing(array, len_input=params['len_input']) for array in real_example ]
            real_example = [ array[ np.random.choice(array.shape[0]) , : , : ] for array in real_example ]
            real_example = [ np.expand_dims(array, axis=0) for array in real_example ]
            real_example = np.concatenate(real_example)

            ### TODO: UNIFORMARE QUESTI INPUT CON LA FUNZIONE SOPRA
            generator_current_loss, discriminator_current_loss = train_step(X_batch, Y_batch, real_example, mask, params['loss_weight'])

            if iteration % 50 == 0:

                # Check Imputer's plain Loss on training example
                generator_imputation = generator(X_batch)
                train_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(generator_imputation, tf.expand_dims(mask, axis=-1)) - tf.math.multiply(Y_batch, tf.expand_dims(mask, axis=-1))))

                # To get Generative and Aversarial Losses (and binary accuracy)
                generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                discriminator_guess_reals = discriminator(real_example)
                discriminator_guess_fakes = discriminator(generator_imputation)

                # Extract X, Y and Mask and stack them in final arrays
                batch = np.random.choice(V_files, size=params['validation_batch_size'], replace=False)
                batch = [ np.load('{}/data_processed/Validation/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]
                batch = [ process_series(array, params) for array in batch ]

                X_batch = [array[0] for array in batch]
                Y_batch = [array[1] for array in batch]
                mask = [array[2] for array in batch]
                X_batch = np.concatenate(X_batch)
                Y_batch = np.concatenate(Y_batch)
                mask = np.concatenate(mask)

                generator_imputation = generator(X_batch)
                val_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(tf.squeeze(generator_imputation), mask) - tf.math.multiply(tf.squeeze(Y_batch), mask)))

                print('{}.{}   \tGenerator Loss: {}   \tDiscriminator Loss: {}   \tDiscriminator Accuracy (reals, fakes): ({}, {})'.format(
                    epoch, iteration,
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes))
                ))
                print('\t\tImputation Loss: {}   \tValidation Loss: {}\n'.format(train_loss, val_loss))

    print('\nTraining complete.\n')

    generator.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Generator saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    if params['save_discriminator']:
        discriminator.save('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name']))
        print('\nDiscriminator saved at:\n{}'.format('{}/saved_models/{}_discriminator.h5'.format(os.getcwd(), params['model_name'])))

    return None
