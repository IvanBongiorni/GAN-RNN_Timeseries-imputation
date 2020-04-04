
def train(model, params, X_train, Y_train, X_val, Y_val):
    import time
    import numpy as np
    import tensorflow as tf

    @tf.function
    def train_on_batch():
        take = iteration * batch_size
        X_batch = X_train[ take:take+batch_size , : ]
        Y_batch = Y_train[ take:take+batch_size , : ]

        with tf.GrandientTape() as tape:
            current_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(Y_batch, model(X_batch),
                                                                from_logits = True))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    for epoch in range(params['n_epochs']):
        start = time.time()

        for iteration in range(X_train.shape[0] // params['batch_size']):
            current_loss = train_on_batch()
            # loss_history.append(current_loss)

        validation_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(Y_val, model(X_val),
                                                            from_logits = True))

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


def train_GAN(generator, discriminator, params):
    import numpy as np
    import tensorflow as tf

    ## Loss functions for Generator and Discriminator
    def generator_loss(fake_output):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    # ## Training function
    # @tf.function
    # def train_on_batch(batch):
    #
    #     noise = tf.random.normal([params['batch_size'], params['noise_dims']])
    #
    #     with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
    #
    #         real_output =
    #         fake_output =
    #
    #     gradients_of_generator = generator_tape.gradient(generator_loss, generator.trainable_variables)
    #     gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    #
    #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    #
    #     return generator_loss, discriminator_loss



    for epoch in range(params['n_epochs']):

        for iteration in range(X.shape[0]//batch_size):
            # train G, freeze D
            generator.trainable = True
            discriminator.trainable = False


            # Take batch
            take = iteration * params['batch_size']
            X_batch = X_train[ take:take+params['batch_size'] , : ]

            # Apply artificial deterioration and impute data
            X_imputed = deterioration.apply(X_batch)
            X_imputed = model(X_imputed)
            

            # Categorical Crossentropy:
            # X_batch with tf.ones_like
            # X_corrupted with tf.zeros_like

            current_G_loss = G_loss(X_batch, X_imputed)


            current_D_loss = train_Discriminator

        for iteration in range(X.shape[0]//batch_size):
            # train A, freeze G
            for layer in A.layers: layer.trainable = True
            for layer in G.layers: layer.trainable = False

            # Take batch
            # Apply artificial deterioration
            # Get G's predictions
            # Pair them with other real observations
            # Train A



    # Test on validation

    return None
