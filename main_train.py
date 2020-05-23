"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-24

Main script - Wrapper of the whole processing+training pipeline

Imports processed data, loads config params, runs training pipeline:
builds model (either vanilla or GAN) and trains it, checks loss on test data.
"""

import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pdb import set_trace as BP


def main():
    ''' Wrapper of Training Pipeline. '''
    import os
    import yaml, pickle
    import numpy as np
    import tensorflow as tf

    # local modules
    import model, train, holdout

    print('Loading configuration parameters.')
    params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader )

    print('Setting GPU configurations.')
    ### Sets GPU configurations
    if params['use_gpu']:
        # This prevents CuDNN 'Failed to get convolution algorithm' error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # To see list of allocated tensors in case of OOM
        tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

    else:
        try:
            # Disable all GPUs
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            print('Invalid device or cannot modify virtual devices once initialized.')
        pass


    print('\nStart Training Pipeline.\n')

    if params['model_type'] == 1:
        # If the model already exists load it, otherwise make a new one
        if params['model_name']+'.h5' in os.listdir( os.getcwd() + '/saved_models/' ):
            print('Loading existing model: {}.'.format(params['model_name']))
            Imputer = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] + '.h5' )
        else:
            print('\nVanilla Seq2seq model instantiated as:\n')
            Imputer = model.build_vanilla_seq2seq(params)
        Imputer.summary()
        print('\nStart training.\n')
        train.train_vanilla_seq2seq(Imputer, params)

    elif params['model_type'] == 2:
        if params['model_name']+'.h5' in os.listdir( os.getcwd() + '/saved_models/' ):
            print('Loading existing model: {}.'.format(params['model_name']))
            Imputer = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] + '.h5' )
            Discriminator = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] + '_discriminator.h5' )
        else:
            print('\nGAN Seq2seq model instantiated as:\n')
            Imputer, Discriminator = model.build_GAN(params)
        Imputer.summary()
        print('\nStart GAN training.\n')
        train.train_GAN(Imputer, Discriminator, params)

    elif params['model_type'] == 3:
        if params['model_name']+'.h5' in os.listdir( os.getcwd() + '/saved_models/' ):
            print('Loading existing model: {}.'.format(params['model_name']))
            Imputer = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] + '.h5' )
            Discriminator = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] + '_discriminator.h5' )
        else:
            print('\nPartially adversarial Seq2seq model instantiated as:\n')
            Imputer, Discriminator = model.build_GAN(params)
        Imputer.summary()
        print('\nStart partially adversarial training.\n')
        train.train_partial_GAN(Imputer, Discriminator, params)

    else:
        print("ERROR:\nIn config.yaml, from 'model_type' parameter, specify one of the following model architectures:")
        print("'model_type': 1\tVanilla Seq2seq")
        print("'model_type': 2\tGAN Seq2seq")
        print("'model_type': 3\tSeq2seq with partially adversarial training")
        quit()

    # Check performance on Validation data and Test, optionally
    holdout.run_test(Imputer,
                     params,
                     check_test_performance = params['check_test_performance'])

    return None


if __name__ == '__main__':
    main()
