"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-24

Main script - Wrapper of the whole processing+training pipeline

Imports processed data, loads config params, runs training pipeline:
builds model (either vanilla or GAN) and trains it, checks loss on test data.
"""

def train_main():
    import os
    import yaml, pickle
    import numpy as np
    import tensorflow as tf

    # local modules
    import model, train

    print('\nStart Training Pipeline.\n')

    print('Loading Train and Validation data and configuration parameters.')
    params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader )
    X = np.load( os.getcwd() + '/data_processed/X_train.npy' )
    V = np.load( os.getcwd() + '/data_processed/X_val.npy' )

    if params['gan']:
        Imputer, Discriminator = model.build_GAN(params)
        print('Model instantiated as:\n')
        Imputer.summary()
        print('\nStart of adversarial training.\n')
        train.train_GAN(Imputer, Discriminator, X, V, params)
    else:
        Imputer = model.build(params)
        print('Model instantiated as:\n')
        Imputer.summary()
        print('\nStart training.\n')
        train.train(Imputer, X, V, params)

    # Check performance on Test data
    print('\nPerformance check on Test data:')
    del X, V  # free memory
    T = np.load( os.getcwd() + '/data_processed/X_train.npy' )

    test_loss = train.chech_performance_on_test_data(Imputer, T)
    print('Test Loss: {}'.format(test_loss))

    return None


if __name__ == '__main__':
    train_main()
