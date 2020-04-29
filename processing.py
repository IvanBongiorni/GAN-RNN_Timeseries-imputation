"""
Author: Ivan Bongiorni
2020-04-27

Data preprocessing pipeline. Separated from model implementation and training.
"""
import tools

def _load_raw_dataset(path_data):
    import numpy as np
    import pandas as pd

    df.drop('Page', axis = 1, inplace = True)
    df = df.values
    return df


def process_page_data(df):
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

    page_data = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_data = pd.concat(page_data, axis = 0)
    page_data.reset_index(drop = True, inplace = True)

    page_data['Page'] = df['Page'].copy()  # Attach 'Page' to page_data for merging
    df.drop('Page', axis = 1, inplace = True)
    return df, page_data


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
    df, page_data = process_page_data(df)

    print('Preprocessing trends by language group.')
    X_train = []
    X_val = []
    X_test = []
    X_nan = []          # This is the remaining observations that contain NaN's and cannot be used
    scaling_dict = {}   # This is to save scaling params - by language subgroup

    for language in languages:
        start = time.time()

        sdf = df[ page_data['language'] == language ].values
        sdf_page_data = page_data[ page_data['language'] == language ].values

        # Fill left-NaN's with zeros and trim right NaN's
        for i in range(sdf.shape[0]):
            sdf[ i , : ] = series
            series = tools.left_zero_fill( series )
            series = tools.right_trim_nan( series )
            sdf[ i , : ] = series

        # Extraction of obs with NaN's - not good for training
        X_nan.append( sdf[ np.isnan(sdf).any(axis = 1) ] )  # take rows with NaN's out
        sdf = sdf[ ~np.isnan(sdf).any(axis = 1) ]  # keep only complete obs for training

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
        scaling_percentile = np.percentile(sdf_train, 99)
        sdf_train = scale_trends(sdf, scaling_percentile)
        sdf_val = scale_trends(sdf_val, scaling_percentile)
        sdf_test = scale_trends(sdf_val, scaling_percentile)
        scaling_dict[language] = scaling_percentile

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

        print("Lanuage '{}' executed in {} ss.".format(language, round(time.time()-start, 2)))


    X_train = np.concatenate(X_train)
    shuffle = np.random.choice(X.shape[0], X.shape[0] replace = False)
    X_train = X_train[ shuffle , : ]
    X_train = pd.DataFrame(X_train)
    X_train.to_pickle()

    X_val = np.concatenate(X_val)
    X_val = pd.DataFrame(X_val)
    X_val.to_pickle()

    X_test = np.concatenate(X_test)
    X_test = pd.DataFrame(X_test)
    X_test.to_pickle()

    # Pickle sub-dataframe with real NaN's to specified folder
    X_nan = np.concatenate(X_nan)
    fileObject = open( os.getcwd + '/data/X_nan.pkl', 'wb')
    pickle.dump(X_nan, fileObject)
    fileObject.close()

    # Save scaling params to file
    yaml.dump(scaling_dict,
              open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'),
              default_flow_style = False)

    return None


if __name__ == '__main__':
    main()
