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
    import pickle
    import numpy as np

    languages = [ 'en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'na' ]

    df = pd.read_csv('{}train_2.csv'.format(path_data))

    print('Extracting URL metadata from dataframe.')
    df, page_data = process_page_data(df)

    print('Preprocessing trends by language group.')
    X = []              # This is the final training dataset
    X_nan = []          # This is the remaining observations that contain NaN's and cannot be used
    scaling_dict = {}   # This is to save scaling params - by language subgroup
    
    for language in languages:
        sdf = df[ page_data['language'] == language ].values
        sdf_page_data = page_data[ page_data['language'] == language ].values

        # Fill left-NaN's with zero
        for i in range(sdf.shape[0]):
            sdf[ i , : ] = tools.left_zero_fill( sdf[ i , : ] )

        X_nan.append( sdf[ np.isnan(sdf).any(axis = 1) ] )  # Take rows with NaN's out

        # Keep only complete observations for training
        sdf = sdf[ ~np.isnan(sdf).any(axis = 1) ]

        sdf = [ tools.right_trim(sdf[ i , : ], params) for i in range(sdf.shape[0]) ]
        sdf = [ tools.RNN_univariate_processing(sdf[ i , : ], params) for i in range(sdf.shape[0]) ]
        sdf = np.concatenate(sdf)

        sdf, scaling_percentile, = scale_trends(sdf, params)
        scaling_dict[language] = scaling_percentile

        X.append(sdf)
        print("\t'{}'.".format(language))

    # Pickle sub-dataframe with real NaN's to specified folder
    X_nan = np.concatenate(X_nan)
    fileObject = open(params['path_data']+'X_nan.pkl', 'wb')
    pickle.dump(X_nan, fileObject)
    fileObject.close()
    del X_nan  # free memory

    # Shuffle and split in Train-Validation-Test based on input params
    # X = shuffle(X, random_state = params['seed'])
    # test_cutoff = int(X.shape[0] * ( 1 - params['val_test_ratio'][0] ))
    # val_cutoff = int(X.shape[0] * ( 1 - np.sum(params['val_test_ratio']) ))
    #
    # V = X[ val_cutoff:test_cutoff , : ]
    # Y = X[ test_cutoff: , : ]
    # X = X[ :val_cutoff , : ]
    return X, V, Y


if __name__ == '__main__':
    main()
