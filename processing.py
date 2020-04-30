"""
Author: Ivan Bongiorni
2020-04-27

Data preprocessing pipeline. Separated from model implementation and training.
"""
import tools


def get_page_language(df):
    """
    Attaches page data to main df by iterating process_url() from
    dataprep_tools.py:
        language:  code - with 'na' for 'no language detected'
        website:   what type of website: 'wikipedia', 'wikimedia', 'mediawiki'
        access:    type of access (e.g.: mobile, desktop, both, ...)
        agent:     type of agent
    """
    import numpy as np
    import pandas as pd
    import tools  # local module

    page_lang = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_lang = pd.Series(page_data)

    df.drop('Page', axis = 1, inplace = True)
    return df, page_lang


def main(params):
    '''
    Main wrapper of the input pipe. Steps:
    1. Loads pandas dataframe, takes values only and converts to np.array
    2. Fills NaN's on the left with zeros, keeps NaN's within trends
    3. Extracts trends with NaN's and pickles them in /data/ folder, leaving
       a dataset of complete trends for training
    4. Shuffles dataset and operates Train-Validation-Test split based on params
    '''
    import os
    import pickle
    import numpy as np
    np.random.seed(params['seed'])

    languages = [ 'en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na' ]

    print('Loading data.')
    current_path = os.getcwd()
    df = pd.read_csv(current_path + '/data/train2.csv')

    print('Extracting URL metadata from dataframe.')
    df, page_lang = get_page_language(df)

    print('Preprocessing trends by language group.')
    X_train = []
    X_val = []
    X_test = []

    scaling_dict = {}   # This is to save scaling params - by language subgroup

    for language in languages:
        start = time.time()

        sdf = df[ page_lang['language'] == language ].values
        sdf_lang = page_lang[ page_lang['language'] == language ].values

        ### SPLIT IN TRAIN - VAL - TEST
        # Generate random index to each row following 'val_test_size' Pr distribution
        sample = np.random.choice(range(3),
        sdf.shape[0],
        p = [ 1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1]],
        replace = True)
        sdf_train = sdf[ sample == 0 ]
        sdf_val = sdf[ sample == 1 ]
        sdf_test = sdf[ sample == 2 ]

        # Scale and save param into dict
        scaling_percentile = np.nanpercentile(sdf_train, 99)  # np.nanpercentile excludes NaN's from computation
        sdf_train = scale_trends(sdf, scaling_percentile)
        sdf_val = scale_trends(sdf_val, scaling_percentile)
        sdf_test = scale_trends(sdf_val, scaling_percentile)
        scaling_dict[language] = scaling_percentile

        # Fill left-NaN's with zeros
        sdf_train = [ tools.left_zero_fill(sdf_train[ i , : ]) for i in range(sdf_train.shape[0]) ]
        sdf_val = [ tools.left_zero_fill(sdf_val[ i , : ]) for i in range(sdf_val.shape[0]) ]
        sdf_test = [ tools.left_zero_fill(sdf_test[ i , : ]) for i in range(sdf_test.shape[0]) ]

        # Trim right-NaN's
        sdf_train = [ tools.right_trim_nan(series, params) for series in sdf_train ]
        sdf_val = [ tools.right_trim_nan(series, params) for series in sdf_val ]
        sdf_test = [ tools.right_trim_nan(series, params) for series in sdf_test ]

        # Exclude trends that still contain internal NaN's
        sdf_train = [ series for series in sdf_train if np.sum(np.isnan(series)) = 0 ]
        sdf_val = [ series for series in sdf_val if np.sum(np.isnan(series)) = 0 ]
        sdf_test = [ series for series in sdf_test if np.sum(np.isnan(series)) = 0 ]

        # Exclude trends that are not long enough to be fed into the series
        sdf_train = [ series for series in sdf_train if len(series) >= params['len_input'] ]
        sdf_val = [ series for series in sdf_val if len(series) >= params['len_input'] ]
        sdf_test = [ series for series in sdf_test if len(series) >= params['len_input'] ]

        # Process to RNN format ('sliding window' to input series) and pack into final array
        sdf_train = [ tools.RNN_univariate_processing(series, params) for series in sdf_train ]
        sdf_train = np.concatenate(sdf_train)

        sdf_val = [ tools.RNN_univariate_processing(series, params) for series in sdf_val ]
        sdf_val = np.concatenate(sdf_val)

        sdf_test = [ tools.RNN_univariate_processing(series, params) for series in sdf_test ]
        sdf_test = np.concatenate(sdf_test)

        X_train.append(sdf_train)
        X_val.append(sdf_val)
        X_test.append(sdf_test)

        print("\tSub-dataframe for language '{}' executed in {} ss.".format(language, round(time.time()-start, 2)))

    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)
    
    # shuffle X_train for batch training
    shuffle = np.random.choice(X.shape[0], X.shape[0] replace = False)
    X_train = X_train[ shuffle , : ]

    pickle.dump(X_train, open( os.getcwd() + '/data_processed/X_train.pkl' ))
    pickle.dump(X_val, open( os.getcwd() + '/data_processed/X_train.pkl' ))
    pickle.dump(X_test, open(os.getcwd() + '/data_processed/X_val.pkl'))

    # Save scaling params to file
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    return None


if __name__ == '__main__':
    main()
