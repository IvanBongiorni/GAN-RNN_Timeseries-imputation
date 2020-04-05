
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

    ## TRAINING FUNCTIONS
    @tf.function
    def train_generator(prediction_imputed):
        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(prediction_imputed), prediction_imputed)

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

        for iteration in range(X.shape[0]//batch_size):

            # Take batch and apply artificial deterioration to impute data
            take = iteration * params['batch_size']
            X_real = X_train[ take:take+params['batch_size'] , : ]
            X_imputed = deterioration.apply(X_real)
            X_imputed = model(X_imputed)


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

            generator_current_loss = train_generator(prediction_imputed)

        print('{} - {}.  \t Generator Loss: {}.  \t Discriminator Loss: {}'.format(
            epoch, iteration,
            generator_current_loss,
            discriminator_current_loss
        ))

    return None
