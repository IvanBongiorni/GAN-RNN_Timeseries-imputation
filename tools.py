"""
Author: Ivan Bongiorni
2020-04-27

Tools for data processing pipeline. These are more technical functions to be iterated during
main pipeline run.
"""
import os
import re
import time
import numpy as np
import pandas as pd

from pdb import set_trace as BP


def set_gpu_configurations(params):
    '''
    Sets GPU configurations, either deactivates it, or allows for GPU memory
    growth in order to avoid "Failed to get convolution algorithm" error.
    '''
    import tensorflow as tf

    print('Setting GPU configurations.')
    ### Sets GPU configurations
    if params['use_gpu']:
        # This prevents CuDNN 'Failed to get convolution algorithm' error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # To see list of allocated tensors in case of OOM
        tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

    else:
        try:
            # Disable all GPUs
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            print('Invalid device or cannot modify virtual devices once initialized.')
        pass
    return None


def left_zero_fill(x):
    import numpy as np
    if np.isfinite(x[0]):
        return x

    cumsum = np.cumsum(np.isnan(x))
    x[ :np.argmax(cumsum[:-1] == cumsum[1:]) + 1] = 0
    return x


def process_url(url):
    """
    Extracts four variables from URL string:
        language:  code - with 'na' for 'no language detected'
        website:   what type of website: 'wikipedia', 'wikimedia', 'mediawiki'
        access:    type of access: 'all-access', 'desktop', 'mobile-web'
        agent:     type of agent: 'spider', 'all-agents'
    """
    import re
    import numpy as np
    import pandas as pd

    if '_en.' in url: language = 'en'
    elif '_ja.' in url: language = 'ja'
    elif '_de.' in url: language = 'de'
    elif '_fr.' in url: language = 'fr'
    elif '_zh.' in url: language = 'zh'
    elif '_ru.' in url: language = 'ru'
    elif '_es.' in url: language = 'es'
    else: language = 'na'

    if 'wikipedia' in url: website = 'wikipedia' #-1
    elif 'wikimedia' in url: website = 'wikimedia' #0
    elif 'mediawiki' in url: website = 'mediawiki' #1

    access, agent = re.split('_', url)[-2:]

    url_features = {
        # 'url': url,
        'language': language,
        'website': website,
        'access': access,
        'agent': agent
    }
    return url_features


def get_time_schema(df):
    """ Returns np.array with patterns for time-related variables (year/week days)
    in [0,1] range, to be repeated on all trends. """
    import numpy as np
    import pandas as pd

    daterange = pd.date_range(df.columns[0], df.columns[-1], freq='D').to_series()

    weekdays = daterange.dt.dayofweek
    weekdays = weekdays.values / weekdays.max()
    yeardays = daterange.dt.dayofyear
    yeardays = yeardays.values / yeardays.max()

    # First year won't enter the Train set because of year lag
    weekdays = weekdays[ 365: ]
    yeardays = yeardays[ 365: ]

    return weekdays, yeardays


def scale_trends(array, scaling_percentile = None):
    """
    Takes a linguistic sub-dataframe and applies a robust custom scaling in two steps:
        1. log( x + 1 )
        2. Robust min-max scaling to [ 0, 99th percentile ]
    If scaling percentile is to be found (Train data), then a scaling percentile value
    is found and both array and percentile are returned for Val and Test scaling.
    """
    import numpy as np

    array = np.log(array + 1)

    if not scaling_percentile:
        scaling_percentile = np.nanpercentile(array, 99)
        array = array / scaling_percentile
        return array, scaling_percentile
    else:
        array = array / scaling_percentile
        return array


def right_trim_nan(x):
    ''' Trims all NaN's on the right '''
    import numpy as np

    if np.isnan(x[-1]):
        cut = np.argmax(np.isfinite(x[::-1]))
        return x[ :-cut ]
    else:
        return x


def apply_processing_transformations(trend, vars, weekdays, yeardays, params):
    '''
    Takes trend and webpage variables and applies pre-processing: left pad and
    right trim NaN's, filters trends of insufficient length.
    Finally generates a 2D array to be stored and loaded during training.
    '''
    import numpy as np
    import tools  # local import

    trend = tools.left_zero_fill(trend) # Fill left-NaN's with zeros
    trend = tools.right_trim_nan(trend) # Trim right-NaN's

    # Exclude trends that still contain internal NaN's or not long enough to be fed into the series
    if np.sum(np.isnan(trend)) > 0 or len(trend) < params['len_input'] + 365:
        return None

    #Combine trend and all other input vars into a 2D array to be stored on drive. '''
    trend_lag_year = np.copy(trend[:-365])
    trend_lag_quarter = np.copy(trend[180:])
    trend = trend[365:]
    trend_lag_quarter = trend_lag_quarter[:len(trend)]

    X = np.column_stack([
        trend,                           # trend
        trend_lag_quarter,               # trend _ 1 quarter lag
        trend_lag_year,                  # trend _ 1 year lag
        np.repeat(vars[0], len(trend)),  # page variable dummies
        np.repeat(vars[1], len(trend)),
        np.repeat(vars[2], len(trend)),
        np.repeat(vars[3], len(trend)),
        np.repeat(vars[4], len(trend)),
        np.repeat(vars[5], len(trend)),
        np.repeat(vars[6], len(trend)),
        np.repeat(vars[7], len(trend)),
        np.repeat(vars[8], len(trend)),
        np.repeat(vars[9], len(trend)),
        np.repeat(vars[10], len(trend)),
        np.repeat(vars[11], len(trend)),
        weekdays[:len(trend)],           # weekday in [0,1]
        yeardays[:len(trend)]            # day of the year in [0,1]
    ])

    X = X.astype(np.float32)
    return X


def RNN_multivariate_processing(array, len_input):
    '''
    Takes a 2D array with trend and associated variables, and turns it into a 3D
    array for RNN with shape:
        ( no. observations , params['len_input'] , no. input vars )
    For each variable, iterates _univariate_processing() internal function, that
    from 1D series creates 2D matrix of sequences defined by params['len_input']
    '''
    import numpy as np

    def _univariate_processing(series, len_input):
        S = [ series[i : i+len_input] for i in range(len(series)-len_input+1) ]
        return np.stack(S)
        
    array = [ _univariate_processing(array[:,i], len_input) for i in range(array.shape[1]) ]
    array = np.dstack(array)
    return array
