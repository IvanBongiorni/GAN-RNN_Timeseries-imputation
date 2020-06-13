"""
Author: Ivan Bongiorni
2020-04-27

Data preprocessing pipeline. Separated from model implementation and training.
"""
import os
import time
from pdb import set_trace as BP

import numpy as np
import pandas as pd

import tools  # local import


def apply_processing_transformations(array, vars, weekdays, yeardays, params):
    '''
    Wrapper of the main data transformations. Since they had to be repeated on
    Train, Validation, Test sets, I packed a funtion to avoid repetitions.
    '''
    import numpy as np
    import tools  # local import

    len_raw_trend = array.shape[1]

    # Fill left-NaN's with zeros
    array = [ tools.left_zero_fill(array[ i , : ]) for i in range(array.shape[0]) ]

    # Trim right-NaN's
    array = [ tools.right_trim_nan(series) for series in array ]

    # Exclude trends that still contain internal NaN's
    # array = [ series for series in array if np.sum(np.isnan(series)) == 0 ]
    array = [ array[i] for i in range(len(aray)) if np.sum(np.isnan(array[i])) == 0 ]
    vars = [ vars[i,:] for i in range(array.shape[0]) if np.sum(np.isnan(array[i])) == 0 ]

    # Exclude trends that are not long enough to be fed into the series
    # array = [ series for series in array if len(series) >= params['len_input'] ]
    array = [ array[i] for i in range(len(array)) if len(array[i]) >= params['len_input'] + 365 ]
    vars = [ vars[i] for i in range(len(array)) if len(array[i]) >= params['len_input'] + 365 ]

    # Refill the NaN's on the right for stacking back to matrix and saving
    array = [ np.concatenate([series, np.repeat(np.nan, len_raw_trend-len(series))]) if len(series) < len_raw_trend else series for series in array ]

    # Create 2D matrices out of every trend + page vars
    array = [ tools.get_training_matrix(trend=element[0], page=element[1], weekdays=weekdays, yeardays=yeardays) for element in zip(array, vars) ]

    return array


def processing_main():
    '''
    Main wrapper of the input pipe. Steps:
    1. Loads pandas dataframe, takes values only and converts to np.array
    2. Fills NaN's on the left with zeros, keeps NaN's within trends
    3. Extracts trends with NaN's and pickles them in /data/ folder, leaving
       a dataset of complete trends for training
    4. Shuffles dataset and operates Train-Validation-Test split based on params
    '''
    import os
    import time
    import yaml
    import pickle
    import numpy as np
    import pandas as pd

    pipeline_start = time.time()

    print('\nStart data processing pipeline.\n')
    print('Loading data and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/train_2.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    print('Extracting URL metadata from dataframe.')
    page_vars = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_vars = pd.DataFrame(page_vars)
    df.drop('Page', axis = 1, inplace = True)
    
    # get fixed time variables
    weekdays, yeardays = tools.get_time_schema(df)

    ### SPLIT IN TRAIN - VAL - TEST
    # np.random.seed(params['seed'])
    # Generate random index to each row following 'val_test_size' Pr distribution
    print('Training - Validation - Test split.')
    sample = np.random.choice(
        range(3),
        sdf.shape[0],
        p = [1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1]],
        replace = True)

    X_train = df[ sample == 0 ]
    page_vars_train = page_vars[ sample == 0 ]
    X_val = df[ sample == 1 ]
    page_vars_val = page_vars[ sample == 1 ]
    X_test = df[ sample == 2 ]
    page_vars_test = page_vars[ sample == 2 ]

    del df, page_vars # free memory

    print('Scaling data.')
    X_train, scaling_percentile = scale_trends(X_train)  # scaling_percentile=None, to get one
    X_val = scale_trends(X_val, scaling_percentile = scaling_percentile)
    X_test = scale_trends(X_test, scaling_percentile = scaling_percentile)

    # Save scaling params to file
    scaling_dict = {'percentile': float(scaling_percentile)}
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    print('Start processing of input variables.')
    # Apply sequence of processing transformations, save to folder, and del sdf's to free memory
    X_train = apply_processing_transformations(X_train, page_vars_train, weekdays, yeardays, params)
    X_val = apply_processing_transformations(X_val, page_vars_val, weekdays, yeardays, params)
    X_test = apply_processing_transformations(X_test, page_vars_test, weekdays, yeardays, params)

    print('Saving {} Training observations.'.format(len(X_train)))
    for i in range(len(X_train)):
        np.save(os.getcwd() + '/data_processed/Training/X_{}'.format(str(i).zfill(6)), X_train[i])

    print('Saving {} Validation observations.'.format(len(X_val)))
    for i in range(len(X_val)):
        np.save(os.getcwd() + '/data_processed/Validation/V_{}'.format(str(i).zfill(6)), X_val[i])

    print('Saving {} Test observations.'.format(len(X_test)))
    for i in range(len(X_test)):
        np.save(os.getcwd() + '/data_processed/Test/T_{}'.format(str(i).zfill(6)), X_test[i])

    print('\nPipeline executed in {} ss.\n'.format(round(time.time()-pipeline_start, 2)))
    return None


if __name__ == '__main__':
    processing_main()
