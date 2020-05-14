"""
Author: Ivan Bongiorni
2020-05-14

Performance check on Test data.

This script is called from main_train.py after training is done and params['check_test_performance']
from config.yaml is set to True. It loads the test set from /data_processed/ folder and returns
a Loss value (MAE).
"""

def check_performance_on_test_data(model):
    import os
    import time
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    # Local
    import train   # process_batch required
    import deterioration

    # Load test data
    print('\tLoading Test data.')
    T = np.load( os.getcwd() + '/data_processed/X_test.npy' )
    print('\tShape: {}'.format(T.shape))

    print('\tStart iteration of model predictions...')
    start = time.time()

    # iterate processing and prediction
    P = []
    for i in range(T.shape[0]):
        t = train.process_batch( T[ i , : ] )
        p = model.predict(t)
        P.append(p)
    P = np.concatenate(P)
    print('Done in {}ss.'.format(round(time.time()-start, 2)))

    loss = tf.keras.losses.MeanAbsoluteError()
    final_loss = loss(T, P)

    print('\n\tFinal MAE Loss: {}'.format(final_loss.numpy()))

    return None
