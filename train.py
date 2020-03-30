
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


def train_GAN():
    
    from tensorflow.keras import backend as K
    
    A_loss = tf.keras.losses.CategoricalCrossentropy()
    
    # SPECIFY PROPER GAN LOSSES
    # G_loss = tf.keras.losses.MeanAbsoluteError()
    @tf.function
    def G_loss():
        return generator_loss
    
    A_loss = tf.keras.losses.CategoricalCrossentropy()
    
    @tf.function
    def train_Generator():
        return current_G_loss
    
    @tf.function
    def train_Adversary():
        return current_A_loss
    
    for epoch in range(params['n_epochs']):
        
        for iteration in range(X.shape[0]//batch_size):
            # train G, freeze A
            for layer in A.layers: layer.trainable = False
            for layer in G.layers: layer.trainable = True
            
            # Take batch
            # Apply artificial deterioration
            # Pair them with actual (uncorrupted) batch
            
            current_G_loss = train_Generator()
            
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
