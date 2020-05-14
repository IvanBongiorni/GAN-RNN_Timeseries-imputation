"""
Author: Ivan Bongiorni
2020-05-14

Performance check on Test data.

This script is called from main_train.py after training is done and params['check_test_data'] 
from config.yaml is set to True. It loads the test set from /data_processed/ folder and returns
a Loss value (MAE).
"""
import os
import time

import numpy as np
import pandas as pd


def check_performance_on_test_data():
    import numpy as np
    import tensorflow as tf

    # Load test data
    print('\tLoading Test data.')
    T = np.load( os.getcwd() + '/data_processed/X_test.npy' )
    print('\tShape: {}'.format(T.shape))

    #
    model = tf.keras.models.load_model( os.getcwd() + '/saved_models/' + params['model_name'] ) 

    return None
