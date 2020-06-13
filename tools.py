"""
Author: Ivan Bongiorni
2020-04-27

Tools for data processing pipeline. These are more technical functions to be iterated during
main pipeline run.
"""
# import numba


# @numba.jit(python = True)
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
        access:    type of access (e.g.: mobile, desktop, both, ...)
        agent:     type of agent
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

    if 'wikipedia' in url: website = 'wikipedia'
    elif 'wikimedia' in url: website = 'wikimedia'
    elif 'mediawiki' in url: website = 'mediawiki'

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
    daterange = pd.date_range(df.columns[1], df.columns[-1], freq='D').to_series()

    weekdays = daterange.dt.dayofweek
    weekdays = weekdays.values / weekdays.max()
    yeardays = daterange.dt.dayofyear
    yeardays = yeardays.values / yeardays.max()

    weekdays = weekdays.values
    yeardays = yeardays.values

    # First year won't enter the Train set because of year lag
    weekdays = weekdays[ 365: ]
    yeardays = yeardays[ 365: ]

    return weekdays, yeardays


# @numba.jit(python = True)
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
        scaling_percentile = np.percentile(array, 99)
        array = array / scaling_percentile
        return array, scaling_percentile
    else:
        array = array / scaling_percentile
        return array


# @numba.jit(python = True)
def right_trim_nan(x):
    """ Trims all NaN's on the right """
    import numpy as np

    if np.isnan(x[-1]):
        cut = np.argmax(np.isfinite(x[::-1]))
        return x[ :-cut ]
    else:
        return x


def get_training_matrix(trend, page, weekdays, yeardays):
    ''' Combines trend and all other input vars into a 2D array to be stored on drive. '''
    import numpy as np

    trend_lag_year = trend[ :-365 ]
    trend_lag_quarter = trend[ 180:-180 ]
    trend = trend[ 365: ]

    X = np.column_stack([
        trend,                           # trend
        trend_lag_quarter,               # trend _ lag 1 quarter
        trend_lag_year,                  # trend _ lag 1 year
        np.repeat(page[0], len(trend)),  # language
        np.repeat(page[1], len(trend)),  # website
        np.repeat(page[2], len(trend)),  # access
        np.repeat(page[3], len(trend)),  # agent
        weekday[:len(trend)],            # weekday in [0,1]
        yearday[:len(trend)]             # day of the year in [0,1]
    ])
    return X


# @numba.jit(python = True)
def RNN_univariate_processing(series, len_input):
    ''' From 1D series creates 2D matrix of sequences defined by params['len_input'] '''
    # This function is a simplification of RNN_dataprep from:
    # https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks/blob/master/TensorFlow2.0__04.02_RNN_many2many.ipynb
    import numpy as np
    import numpy as np

    S = [ series[i : i+len_input] for i in range(len(series)-len_input+1) ]
    S = np.stack(S)

    S = S.astype(np.float32)
    return S
