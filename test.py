"""
Author: Ivan Bongiorni
2020-05-14

Performance check on Test data.

This script is called from main_train.py after training is done and params['check_test_performance']
from config.yaml is set to True.
"""

def check(model, return_stats = False):
    '''
    Loads Test observations from /data_processed/ folder, iterates model
    prediction and return a Loss value (MAE) and some error statistics.

    Argument `return_stats` is meant to be used only from Jupyter Notebooks
    for more thorough visualization and analyses.
    '''
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

    # return some error statistics here

    #min
    #25 perc
    #mean
    #median
    #75 perc
    #max

    if return_stats:

        plt.figure(figsize = (15, 7))
        plt.hist(E, bins = 100)
        plt.title('Test errors of model: {} (MAE)'.format(params['model_name']))
        plt.show()

        error_stats = {
            'min': error_min,
            '25_perc': error_25p,
            'mean' error_mean,
            'median': error_median,
            '75_perc': error_75p,
            'max' error_max
            }
        return error_stats

    else:
        return None
