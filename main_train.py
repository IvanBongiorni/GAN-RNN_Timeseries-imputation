"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-24

Main script - Wrapper of the whole processing+training pipeline

Imports processed data, loads config params, runs training pipeline:
builds model (either vanilla or GAN) and trains it, checks loss on test data.
"""

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pdb import set_trace as BP


def train_main():
    import os
    import yaml, pickle
    import numpy as np
    import tensorflow as tf

    # local modules
    import model, train

    print('Loading Train and Validation data and configuration parameters:')
    params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader )

    # Set GPU configurations
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

    X = np.load( os.getcwd() + '/data_processed/X_train.npy' )
    V = np.load( os.getcwd() + '/data_processed/X_val.npy' )
    print('Train set:      ', X.shape)
    print('Validation set: ', V.shape, '\n')

    print('\nModel instantiated as:\n')
    if params['gan']:
        Imputer, Discriminator = model.build_GAN(params)
        Imputer.summary()
        print('\nStart of adversarial training.\n')
        train.train_GAN(Imputer, Discriminator, X, V, params)
    else:
        Imputer = model.build(params)
        Imputer.summary()
        print('\nStart training.\n')
        train.train(Imputer, X, V, params)

    # Check performance on Test data
    print('\nPerformance check on Test data:')
    del X, V  # free memory

    if params['check_test_performance']:
        test_loss = train.chech_performance_on_test_data(Imputer)
        print('\nChecking performance on Test data')

    return None


if __name__ == '__main__':
    train_main()
