"""
Author: Ivan Bongiorni
2020-04-27

Tools for data processing pipeline. These are more technical functions to be iterated during
main pipeline run.
"""

def left_zero_fill(x):
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

    url_features = pd.DataFrame({
        'language': [language],
        'website': [website],
        'access': [access],
        'agent': [agent]
    })
    return url_features


def RNN_univariate_processing(series, params):
    ''' From 1D series creates 2D matrix of sequences defined by params['len_input'] '''
    # https://github.com/IvanBongiorni/TensorFlow2.0_Notebooks/blob/master/TensorFlow2.0__04.02_RNN_many2many.ipynb
    import numpy as np

    S = [ series[ i : i + params['len_input'] + len_pred] for i in range(S.shape[0]) ]
    S = np.concatenate(S)

    train = S[: , :len_input]
    test = S[: , -len_pred:]

    # reshape data as required by Keras LSTM
    train = train.reshape((len(train), len_input, 1))
    test = test.reshape((len(test), len_pred))

    train = train.astype(np.float32)
    test = test.astype(np.float32)
    return train, test
