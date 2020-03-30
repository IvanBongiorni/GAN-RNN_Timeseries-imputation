"""
This local module contains loading a preprocessing steps for
"""


def _load_data(path_data):
    import numpy as np
    import pandas as pd
    df = pd.read_csv('{}train_2.csv'.format(path_data))
    df.drop('Page', axis = 1, inplace = True)
    df = df.values
    return df


def _left_zero_fill(x):
    if np.isfinite(x[0]):
        return x

    cumsum = np.cumsum(np.isnan(x))
    x[ :np.argmax(cumsum[:-1] == cumsum[1:]) + 1] = 0
    return x


def load_and_process_data(params):
    '''
    Main wrapper of the input pipe. Steps:
    1. Loads pandas dataframe, takes values only and converts to np.array
    2. Fills NaN's on the left with zeros, keeps NaN's within trends
    3. Extracts trends with NaN's and pickles them in /data/ folder, leaving
       a dataset of complete trends for training
    4. Shuffles dataset and operates Train-Validation-Test split based on params
    '''
    import pickle
    import numpy as np

    X = _load_data(params['path_data'])
    
    # Fill left-NaN's with zero
    for i in range(df.shape[0]):
        X[ i , : ] = _left_zero_fill( X[ i , : ] )

    # Separate rows, complete and with NaN's
    X_nan = X[ np.isnan(X).any(axis = 1) ]
    X = X[ ~np.isnan(X).any(axis = 1) ]

    # Pickle X_nan in /data/ folder
    fileObject = open(params['path_data']+'X_nan.pkl', 'wb')
    pickle.dump(df_nan, fileObject)
    fileObject.close()
    
    # Shuffle and split in Train-Validation-Test based on input params
    X = shuffle(X, random_state = params['seed'])
    test_cutoff = int(X.shape[0] * ( 1 - params['val_test_ratio'][0] ))
    val_cutoff = int(X.shape[0] * ( 1 - np.sum(params['val_test_ratio']) ))
    V = X[ val_cutoff:test_cutoff , : ]
    Y = X[ test_cutoff: , : ]
    X = X[ :val_cutoff , : ]
    return X, V, Y
