"""
Author: Ivan Bongiorni
2020-04-27

Data preprocessing pipeline. Separated from model implementation and training.
"""
import tools


# def apply_processing_transformations(array, params, scaling_percentile):
#     '''
#     Wrapper of the main data transformations. Since they had to be repeated on
#     Train, Validation, Test sets, I packed a funtion to avoid repetitions.
#     '''
#     import numpy as np
#     import tools  # local module
#
#     # Scale trends
#     array = tools.scale_trends(array, scaling_percentile)
#
#     # Fill left-NaN's with zeros
#     array = [ tools.left_zero_fill(array[ i , : ]) for i in range(array.shape[0]) ]
#
#     # Trim right-NaN's
#     array = [ tools.right_trim_nan(series) for series in array ]
#
#     # Exclude trends that still contain internal NaN's
#     array = [ series for series in array if np.sum(np.isnan(series)) == 0 ]
#
#     # Exclude trends that are not long enough to be fed into the series
#     array = [ series for series in array if len(series) >= params['len_input'] ]
#
#     # Process to RNN format ('sliding window' to input series) and pack into final array
#     array = [ tools.RNN_univariate_processing(series, len_input = params['len_input']) for series in array ]
#
#     array = np.concatenate(array)
#     return array


def assemble_dataframe(languages, data):
    '''
    This function is due to memory problems, not all dataframes can be loaded in
    RAM at the same time.

    Loads all preprocessed sub-dataframes and compacts them into one final array.
    It must repeated three times, for Train, Validation and Test matrices.
    Arg 'data' must be either 'train', 'val', or 'test'
    '''
    import os
    import pickle
    import numpy as np

    X_final = []

    for language in languages:
        sdf_lang = np.load(os.getcwd() + '/data_processed/sdf_{}_{}.npy'.format(language, data),
                           allow_pickle = True)
        X_final.append(sdf_lang)
        del sdf_lang

    X_final = np.concatenate(X_final)
    return X_final


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

    languages = [ 'en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na' ]

    print('\nStart data processing pipeline.\n')
    print('Loading data and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/train_2.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    print('Extracting URL metadata from dataframe.')
    page_lang = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_lang = pd.Series(page_lang)
    df.drop('Page', axis = 1, inplace = True)

    print('Preprocessing trends by language group.')
    np.random.seed(params['seed'])

    scaling_dict = {}   # This is to save scaling params - by language subgroup

    for language in languages:
        start = time.time()

        sdf = df[ page_lang == language ].values

        ### SPLIT IN TRAIN - VAL - TEST
        # Generate random index to each row following 'val_test_size' Pr distribution
        sample = np.random.choice(range(3),
        sdf.shape[0],
        p = [ 1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1]],
        replace = True)
        sdf_train = sdf[ sample == 0 ]
        sdf_val = sdf[ sample == 1 ]
        sdf_test = sdf[ sample == 2 ]
        del sdf # free memory

        ### I MUST TRANSFER PROCESSING PIPELINE ON SCRATCH, DURING TRAINING

        # Scale and save param into dict
        scaling_percentile = np.nanpercentile(sdf_train, 99)  # np.nanpercentile ignores NaN's
        scaling_dict[language] = float(scaling_percentile)

        # Apply sequence of processing transformations, save to folder, and del sdf's to free memory
        # sdf_train = apply_processing_transformations(sdf_train, params, scaling_percentile)
        # np.save(os.getcwd() + '/data_processed/sdf_{}_train'.format(language), sdf_train)
        # del sdf_train

        # sdf_val = apply_processing_transformations(sdf_val, params, scaling_percentile)
        # np.save(os.getcwd() + '/data_processed/sdf_{}_val'.format(language), sdf_val)
        # del sdf_val

        # sdf_test = apply_processing_transformations(sdf_test, params, scaling_percentile)
        # np.save(os.getcwd() + '/data_processed/sdf_{}_test'.format(language), sdf_test)
        # del sdf_test

        print("\tSub-dataframe for language '{}' executed in {} ss.".format(
            language, round(time.time()-start, 2)))

    # Save scaling params to file
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    # print('Assembling Train, Validation, Test datasets.')
    # # Assemble and save datasets, free memory, delete temporary sdf files
    # X_train = assemble_dataframe(languages, 'train')
    #
    # Shuffle X_train only before training
    # shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], replace = False)
    # X_train = X_train[ shuffle , : ]
    # np.save(os.getcwd() + '/data_processed/X_train', X_train)
    # del X_train
    # for language in languages:
    #     os.remove(os.getcwd() + '/data_processed/sdf_{}_train.npy'.format(language))
    #
    # X_val = assemble_dataframe(languages, 'val')
    # np.save(os.getcwd() + '/data_processed/X_val', X_val)
    # del X_val
    # for language in languages:
    #     os.remove(os.getcwd() + '/data_processed/sdf_{}_train.npy'.format(language))
    #
    # X_test = assemble_dataframe(languages, 'test')
    # np.save(os.getcwd() + '/data_processed/X_test', X_test)
    # del X_test
    # for language in languages:
    #     os.remove(os.getcwd() + '/data_processed/sdf_{}_test.npy'.format(language))

    print('Processed datasets saved at:\n{}'.format(os.getcwd() + '/data_processed/'))
    return None


if __name__ == '__main__':
    processing_main()
