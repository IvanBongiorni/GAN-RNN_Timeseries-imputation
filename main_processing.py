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

import tools


def apply_processing_transformations(array, params, scaling_percentile):
    '''
    Wrapper of the main data transformations. Since they had to be repeated on
    Train, Validation, Test sets, I packed a funtion to avoid repetitions.
    '''
    import numpy as np
    import tools  # local module

    len_raw_trend = array.shape[1]

    # Scale trends
    array = tools.scale_trends(array, scaling_percentile)

    # Fill left-NaN's with zeros
    array = [ tools.left_zero_fill(array[ i , : ]) for i in range(array.shape[0]) ]

    # Trim right-NaN's
    array = [ tools.right_trim_nan(series) for series in array ]

    # Exclude trends that still contain internal NaN's
    array = [ series for series in array if np.sum(np.isnan(series)) == 0 ]

    # Exclude trends that are not long enough to be fed into the series
    array = [ series for series in array if len(series) >= params['len_input'] ]

    # Refill the NaN's on the right for stacking back to matrix and saving
    array = [ np.concatenate([series, np.repeat(np.nan, len_raw_trend-len(series))]) if len(series) < len_raw_trend else series for series in array ]
    array = np.stack(array)

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

    languages = [ 'en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na' ]

    print('\nStart data processing pipeline.\n')
    print('Loading data and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/train_2.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    print('Extracting URL metadata from dataframe.')
    page_lang = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_lang = pd.Series(page_lang)
    df.drop('Page', axis = 1, inplace = True)

    print('Preprocessing trends by language group:')
    np.random.seed(params['seed'])

    X_train = []
    X_val = []
    X_test = []
    scaling_dict = {}   # This is to save scaling params - by language subgroup

    for language in languages:
        start = time.time()

        sdf = df[ page_lang == language ].values

        ### SPLIT IN TRAIN - VAL - TEST
        # Generate random index to each row following 'val_test_size' Pr distribution
        sample = np.random.choice(range(3),
                                  sdf.shape[0],
                                  p = [1-np.sum(params['val_test_ratio']),
                                       params['val_test_ratio'][0],
                                       params['val_test_ratio'][1]],
                                  replace = True)
        sdf_train = sdf[ sample == 0 ]
        sdf_val = sdf[ sample == 1 ]
        sdf_test = sdf[ sample == 2 ]
        del sdf # free memory

        # Scale and save param into dict
        scaling_percentile = np.nanpercentile(sdf_train, 99)  # np.nanpercentile ignores NaN's
        scaling_dict[language] = float(scaling_percentile)

        # Apply sequence of processing transformations, save to folder, and del sdf's to free memory
        sdf_train = apply_processing_transformations(sdf_train, params, scaling_percentile)
        X_train.append(sdf_train)
        del sdf_train

        sdf_val = apply_processing_transformations(sdf_val, params, scaling_percentile)
        X_val.append(sdf_val)
        del sdf_val

        sdf_test = apply_processing_transformations(sdf_test, params, scaling_percentile)
        X_test.append(sdf_test)
        del sdf_test

        print("\tSub-dataframe for language '{}' executed in {} ss.".format(
            language, round(time.time()-start, 2)))

    print('Saving Training data.')
    X_train = np.concatenate(X_train)
    # Shuffle X_train only, before training
    shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], replace = False)
    X_train = X_train[ shuffle , : ]
    for i in range(X_train.shape[0]):
        np.save(os.getcwd() + '/data_processed/Training/X_{}'.format(str(i).zfill(6)), X_train[ i , : ])

    print('Saving Validation data.')
    X_val = np.concatenate(X_val)
    for i in range(X_val.shape[0]):
        np.save(os.getcwd() + '/data_processed/Validation/V_{}'.format(str(i).zfill(6)), X_val[ i , :])

    print('Saving Test data.')
    X_test = np.concatenate(X_test)
    for i in range(X_test.shape[0]):
        np.save(os.getcwd() + '/data_processed/Test/T_{}'.format(str(i).zfill(6)), X_test[ i , : ])

    # Save scaling params to file
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    print('\nPipeline executed in {} ss.\n'.format(round(time.time()-pipeline_start, 2)))
    return None


if __name__ == '__main__':
    processing_main()
