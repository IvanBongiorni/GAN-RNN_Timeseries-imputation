"""
This script loads pre-trained model and imputes missing data from Wikipedia Web Traffic Forecast Dataset.

It is not part of the pipeline like other scripts, it's a practical application of the trained, final model.
"""



def impute():
    import os
    import yaml
    import pickle
    import time

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    # Get current path and update params dict
    current_path = os.getcwd()
    params = yaml.load(open(current_path + '/config.yaml'), yaml.Loader)
    params['data_path'] = current_path + '/data/'
    params['save_path'] = current_path + '/saved_models/'

    print('Loading and extraction of incomplete raw data')
    X = pd.read_csv(load_path)
    X = imputed_data.values
    X = X[ np.isnan(X).any(axis = 1)]  # keep rows with NaN's only

    print('Loading TensorFlow model')
    model = tf.keras.models.load_model(params['save_path'] + params['model_name'])
    imputed_data = model.predict(df)

    print('Saving imputed data...')
    imputed_data = pd.DataFrame(imputed_data)
    imputed_data.to_pickle(save_path + 'imputed_data.pkl')

    print('... done to: {}'.format(save_path))
    return None





if __name__ == '__main__':
    impute()
