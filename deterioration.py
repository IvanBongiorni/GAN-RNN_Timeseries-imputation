
def exponential_noise(x, exp_scale):
    ''' Samples the number of NaN's in x from Exponential distribution; 
    determines their position in the trend; masks actual trend. '''
    import numpy as np
    
    # determine number of NaN's and their position
    nan_number = round(np.random.exponential(scale = exp_scale))
    # nan_draw = np.random.choice([0, 1], size = len(x)) 
    nan_idx = np.random.randint(low = 0, high = len(x), size = nan_number)
    
    x[ nan_idx ] = np.nan
    return x


def blank_random_interval(x, min_size, max_size):
    ''' Draws a blank interval size and its position. '''
    import scipy
    width = int(len(x) * np.random.uniform(min_size, max_size))
    where = np.random.randint(low = 0, high = len(x)-width)
    x[ where:where+width ] = np.nan
    return x


def apply(x, params):
    '''
    Applies random artificial deteriorations. The first coin flip
    determines application of random noise. The second determines 
    the addition of a blank interval. Else applies both.
    '''
    
    if np.random.choice([0, 1], p = [1-params['prob_noise'], params['prob_noise']]):
        x = exponential_noise(x, exp_scale = params['exp_scale'])
    
    elif np.random.choice([0, 1], p = [1-params['prob_interval'], params['prob_interval']]):
        x = blank_random_interval(x, 
                                  min_size = params['interval_ratio'][0], 
                                  max_size = params['interval_ratio'][1])
    else: 
        x = exponential_noise(x, exp_scale = params['exp_scale'])
        x = blank_random_interval(x, 
                                  min_size = params['interval_ratio'][0], 
                                  max_size = params['interval_ratio'][1])
    return x