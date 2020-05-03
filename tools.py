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
    """ Extracts 'language' variable from URL string - 'na' for 'no language detected' """
    import re
    import numpy as np
    import pandas as pd

    if '_en.' in url: lang = 'en'
    elif '_ja.' in url: lang = 'ja'
    elif '_de.' in url: lang = 'de'
    elif '_fr.' in url: lang = 'fr'
    elif '_zh.' in url: lang = 'zh'
    elif '_ru.' in url: lang = 'ru'
    elif '_es.' in url: lang = 'es'
    else: lang = 'na'
    return lang


# @numba.jit(python = True)
def scale_trends(x, scaling_percentile):
    """
    Takes a linguistic sub-dataframe and applies a robust custom scaling in two steps:
        1. log( x + 1 )
        2. Robust min-max scaling to [ 0, 99th percentile ]
    Returns scaled sub-df and scaling percentile, to be saved later in scaling dict
    """
    import numpy as np

    x = np.log(x + 1)
    x = x / np.log(scaling_percentile+1)
    return x


# @numba.jit(python = True)
def right_trim_nan(x):
    """ Trims all NaN's on the right """
    import numpy as np

    if np.isnan(x[-1]):
        cut = np.argmax(np.isfinite(x[::-1]))
        return x[ :-cut ]
    else:
        return x


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
