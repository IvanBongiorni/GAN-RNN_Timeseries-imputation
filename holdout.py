"""
Author: Ivan Bongiorni
2020-05-14

Performance check on Test data.

This script ca be called from main_train.py after training is done and
params['check_test_performance'] from config.yaml is set to True, or from
Jupyter Notebook visualize_imputation.ipynb to check Test performance.
"""

def run_test(model, params, return_stats = False):
    '''
    Loads Test observations from /data_processed/Test/ directory, loads config params
    and trained model. Iterates model prediction to return a final Loss value (MAE)
    error statistics.

    Argument `return_stats` is meant to be used from Jupyter Notebooks for more thorough
    visualization and analyses, it plots a histogram of error distribution, and returns
    a dictionary with error statistics. If set to False error statistics are simply print
    to terminal.
    '''
    import os
    import time
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error as MAE
    import matplotlib.pyplot as plt
    from pdb import set_trace as BP
    import train  #local import


    # Load test data
    print('\tLoading Test data.')
    T = np.load( os.getcwd() + '/data_processed/X_test.npy' )
    print('\tShape: {}'.format(T.shape))

    print('\n\tStart iteration of model predictions...')
    start = time.time()
    
    X = []  # processed trends array
    D = []  # deteriorated trends array
    P = []  # predictions array
    
    # iterate processing and prediction
    for i in range(T.shape[0]):
        # get processed trend and its deteriorated version
        x, d = train.process_batch(T[ i , : ], params)
        X.append(x)
        D.append(d)
        
        p = model.predict(x)
        P.append(p)
    
    X = np.concatenate(X)
    D = np.concatenate(D)
    P = np.concatenate(P)
    print('\tDone in {}ss.'.format(round(time.time()-start, 2)))
    
#     BP()

    loss = tf.keras.losses.MeanAbsoluteError()
    final_loss = loss(X, P)
    print('\n\tFinal MAE Loss: {}'.format(final_loss.numpy()))

    errors = [ MAE( X[ i,:], P[i,:] ) for i in range(X.shape[0]) ]

    error_min = np.min(errors)
    error_25p = np.percentile(errors, 25)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_median = np.median(errors)
    error_75p = np.percentile(errors, 75)
    error_max = np.max(errors)

    if return_stats:
        plt.figure(figsize = (15, 7))
        plt.hist(errors, bins = 100)
        plt.title('Test errors of model: {} (MAE)'.format(params['model_name']))
        plt.show()

        error_stats = {
            'min': error_min,
            '25_perc': error_25p,
            'mean': error_mean,
            'std': error_std,
            'median': error_median,
            '75_perc': error_75p,
            'max': error_max
            }
        return X, D, P, errors, error_stats
    else:
        print('\nError statistics:')
        print('\tMean:', error_mean)
        print('\tSt dev:', error_std, '\n')

        print('\tMin:            ', error_min)
        print('\t25th percentile:', error_25p)
        print('\tMedian:         ', error_median)
        print('\t75th percentile:', error_75p)
        print('\tMax:', error_max)
        return None
