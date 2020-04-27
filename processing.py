"""
This local module contains loading a preprocessing steps for
"""


def _load_raw_dataset(path_data):
    import numpy as np
    import pandas as pd

    df.drop('Page', axis = 1, inplace = True)
    df = df.values
    return df


def _left_zero_fill(x):
    if np.isfinite(x[0]):
        return x

    cumsum = np.cumsum(np.isnan(x))
    x[ :np.argmax(cumsum[:-1] == cumsum[1:]) + 1] = 0
    return x


def RNN_univariate_processing(X, params):
# def RNN_dataprep(series, len_input, len_pred):
    '''
    from: https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks/blob/master/TensorFlow2.0__04.02_RNN_many2many.ipynb

    From time series and two hyperparameters:
    input length and output length, returns Train
    and Test numpy arrays for many-to-many RNNs.

    Args:
        series: time series data
        len_input: length of input sequences
        len_pred: no. of steps ahead to forecast
    '''
    import numpy as np

    # create a matrix of sequences
    S = np.empty((len(series)-(len_input+len_pred)+1,
                  len_input+len_pred))

    # take each row/time window
    for i in range(S.shape[0]):
        S[i,:] = series[i : i+len_input+len_pred]

    # first (len_input) cols of S are train
    train = S[: , :len_input]

    # last (len_pred) cols of S are test
    test = S[: , -len_pred:]

    # set common data type
    train = train.astype(np.float32)
    test = test.astype(np.float32)

    # reshape data as required by Keras LSTM
    train = train.reshape((len(train), len_input, 1))
    test = test.reshape((len(test), len_pred))

    return train, test




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

    df = pd.read_csv('{}train_2.csv'.format(path_data))

    ####
    #### IMPORTANTE: bisogna cambiare tutto, non va bene inserire subito un np.array
    #### perché prima bisogna effettuare una scalatura del dato.
    ####    QUESTA PARTE VA RISCRITTA DEL TUTTO

    X = _load_raw_dataset(params['path_data'])

    # Fill left-NaN's with zero
    for i in range(df.shape[0]):
        X[ i , : ] = _left_zero_fill( X[ i , : ] )

    # Take rows with NaN's out - pickle to folder
    X_nan = X[ np.isnan(X).any(axis = 1) ]
    fileObject = open(params['path_data']+'X_nan.pkl', 'wb')
    pickle.dump(X_nan, fileObject)
    fileObject.close()
    del X_nan  # free memory

    # Keep only complete observations for training
    X = X[ ~np.isnan(X).any(axis = 1) ]


    ###
    ###  INSERIRE RNN_dataprep()
    ###  qualcosa del tipo:  X = RNN_dataprep(X, params)
    X = RNN_univariate_processing(X, params)


    # Shuffle and split in Train-Validation-Test based on input params
    X = shuffle(X, random_state = params['seed'])
    test_cutoff = int(X.shape[0] * ( 1 - params['val_test_ratio'][0] ))
    val_cutoff = int(X.shape[0] * ( 1 - np.sum(params['val_test_ratio']) ))
    V = X[ val_cutoff:test_cutoff , : ]
    Y = X[ test_cutoff: , : ]
    X = X[ :val_cutoff , : ]
    return X, V, Y
