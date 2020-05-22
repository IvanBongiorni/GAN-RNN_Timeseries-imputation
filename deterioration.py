"""
Author: Ivan Bongiorni
2020-05-02

Deterioration function, must be iterated one series at a time. Placeholder values
from params dict will be fed in the training pipeline.
"""
import numpy as np


def _exponential_noise(x, nan_number):
    import numpy as np
    nan_idx = np.random.randint(low = 0, high = len(x), size = nan_number)
    x[ nan_idx ] = np.nan
    return x


def _blank_random_interval(x, min_size, max_size):
    import numpy as np
    width = np.random.randint(min_size, max_size)
    where = np.random.randint(0, len(x)-width)
    x[ where:where+width ] = np.nan
    return x




def apply(X, params):
    '''
    Iterates apply_on_series() on all rows of input numpy 2D array.

    Toss a coin to choose between two options:
    1. Apply exponential noise: draw a number of NaN's to generate. If it's outside
        the rage specified in config.yaml at 'total_nan_range', then apply the option 2.
    2. Blank an interval of the series: sample width and position of an interval in
        the series and turn all points within into NaN.
    '''
    import numpy as np

    threshold_low = int( X.shape[1] * params['total_nan_range'][0] )
    threshold_upp = int( X.shape[1] * params['total_nan_range'][1] )

    def apply_on_series(x):
        if np.random.choice([0, 1], p = [1-params['prob_noise'], params['prob_noise']]):
            nan_number = np.random.randint(low = threshold_low, high = threshold_upp)
            nan_idx = np.random.randint(low = 0, high = len(x), size = nan_number)
            x[ nan_idx ] = np.nan
        else:
            width = np.random.randint(threshold_low, threshold_upp)
            where = np.random.randint(0, len(x)-width)
            x[ where:where+width ] = np.nan
        return x

    X = np.apply_along_axis(apply_on_series, 1, X)
    return X

    # def apply_on_series(x):
    #     if np.random.choice([0, 1], p = [1-params['prob_noise'], params['prob_noise']]):
    #         tot_exp_nan = round(np.random.exponential(scale = params['exp_scale']))
    #
    #         if threshold_low <= tot_exp_nan <= threshold_upp:
    #             x = _exponential_noise(x, tot_exp_nan)
    #         else:
    #             x = _blank_random_interval(x, threshold_low, threshold_upp)
    #     else:
    #         x = _blank_random_interval(x, threshold_low, threshold_upp)
    #     return x
