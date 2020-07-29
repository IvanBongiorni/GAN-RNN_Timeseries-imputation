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

    # One-Hot encode, and leave one out to reduce matrix sparsity
    page_vars = pd.get_dummies(page_vars)
    page_vars.drop(['language_na', 'website_mediawiki', 'access_desktop', 'agent_spider'], axis=1, inplace=True)

    weekdays, yeardays = tools.get_time_schema(df)  # get fixed time variables

    df = df.values
    page_vars = page_vars.values

    ### SPLIT IN TRAIN - VAL - TEST
    # np.random.seed(params['seed'])
    # Generate random index to each row following 'val_test_size' Pr distribution
    print('Training-Validation-Test split.')
    sample = np.random.choice(
        range(3),
        df.shape[0],
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
    X_train, scaling_percentile = tools.scale_trends(X_train)  # scaling_percentile=None, to get one
    X_val = tools.scale_trends(X_val, scaling_percentile = scaling_percentile)
    X_test = tools.scale_trends(X_test, scaling_percentile = scaling_percentile)

    # Save scaling params to file
    scaling_dict = {'percentile': float(scaling_percentile)}
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    print('Start processing of input variables.')
    # Apply sequence of processing transformations and save to folder
    for i in range(X_train.shape[0]):
        array = tools.apply_processing_transformations(
            trend = X_train[i,:],
            vars = page_vars_train[i,:],
            weekdays = weekdays,
            yeardays = yeardays,
            params = params)
        if array is not None:
            np.save(os.getcwd() + '/data_processed/Training/X_{}'.format(str(i).zfill(6)), array)
    print('\tSaved {} Training observations.'.format(X_train.shape[0]))

    for i in range(X_val.shape[0]):
        array = tools.apply_processing_transformations(
            trend = X_val[i,:],
            vars = page_vars_val[i,:],
            weekdays = weekdays,
            yeardays = yeardays,
            params = params)
        if array is not None:
            np.save(os.getcwd() + '/data_processed/Validation/V_{}'.format(str(i).zfill(6)), array)
    print('\tSaved {} Validation observations.'.format(X_val.shape[0]))

    for i in range(X_test.shape[0]):
        array = tools.apply_processing_transformations(
            trend = X_test[i,:],
            vars = page_vars_test[i,:],
            weekdays = weekdays,
            yeardays = yeardays,
            params = params)
        if array is not None:
            np.save(os.getcwd() + '/data_processed/Test/T_{}'.format(str(i).zfill(6)), array)
    print('\tSaved {} Test observations.'.format(X_test.shape[0]))

    print('\nPipeline executed in {} ss.\n'.format(round(time.time()-pipeline_start, 2)))
    return None


if __name__ == '__main__':
    processing_main()
