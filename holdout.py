"""
Author: Ivan Bongiorni
2020-05-14

Performance check on Test data.

This script ca be called from main_train.py after training is done and
params['check_test_performance'] from config.yaml is set to True, or from
Jupyter Notebook visualize_imputation.ipynb to check Test performance.
"""
from pdb import set_trace as BP


def check_performance(model, path, params):
    import os
    import time
    import numpy as np
    # import tensorflow as tf
    from sklearn.metrics import mean_absolute_error as MAE
    import train #local

    start = time.time()

    X = []  # input batch with deteriorated trend
    Y = []  # true trend to be reconstructed
    P = []  # predictions array

    filenames = os.listdir(path)
    if 'readme_validation.md' in filenames: filenames.remove('readme_validation.md')
    if 'readme_test.md' in filenames: filenames.remove('readme_test.md')
    if '.gitignore' in filenames: filenames.remove('.gitignore')

    # Iterate prediction, then pack all vectors in 2D arrays
    for file in filenames:
        batch = np.load(path+file)
        X_batch, Y_batch, _ = train.process_series(batch, params)
        p = model.predict(X_batch)
        X.append(X_batch)
        Y.append(Y_batch)
        P.append(p)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    P = np.concatenate(P)

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    P = np.squeeze(P)
    print('Done in {}ss.'.format(round(time.time()-start, 2)))

    final_loss = MAE(Y, P)
    print('\n\tMAE Loss: {}'.format(final_loss))

    errors = [ MAE(Y[i,:], P[i,:]) for i in range(Y.shape[0]) ]

    return X, Y, P, errors


def get_error_stats(errors, return_dict):
    import numpy as np

    error_min = np.min(errors)
    error_25p = np.percentile(errors, 25)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_median = np.median(errors)
    error_75p = np.percentile(errors, 75)
    error_max = np.max(errors)

    print('\nError statistics:')
    print('\tMean:           ', error_mean)
    print('\tSt dev:         ', error_std, '\n')
    print('\tMin:            ', error_min)
    print('\t25th percentile:', error_25p)
    print('\tMedian:         ', error_median)
    print('\t75th percentile:', error_75p)
    print('\tMax:            ', error_max)

    if return_dict:
        error_stats = {
            'min': error_min,
            '25_perc': error_25p,
            'mean': error_mean,
            'std': error_std,
            'median': error_median,
            '75_perc': error_75p,
            'max': error_max
            }
        return error_stats
    else:
        return None


def run_test(model, params, check_test_performance = False, return_stats = False):
    '''
    First, checks model performance on whole Validation set, and returns a set of
    error stats.

    Second, if check_test_performance == True, runs the same error checks on Test
    set,  but this time also returns processed data (X), deteriorated trends (D),
    and model predictions (P) for inspection in Jupyter Notebooks.
    If running in terminal from main_train.py check_test_performance is set to
    False and only Validation.

    return_stats is meant to be
    '''
    import os
    import time
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import train  #local import

    # Load test data
    print('\nCheck model performance on Validation data.')

    path = os.getcwd() + '/data_processed/Validation/'
    _, _, _, errors = check_performance(model=model, path=path, params=params)
    get_error_stats(errors, return_dict=False)

    if check_test_performance:
        print('\n\nCheck model performance on Test data.')
        filenames = os.listdir( os.getcwd() + '/data_processed/Test/' )
        X, Y, P, errors = check_performance(model=model, path=path, params=params)
        error_stats = get_error_stats(errors, return_dict=True)

        plt.figure(figsize=(15,5))
        plt.hist(errors, bins=100)
        plt.title('Test errors of model: {} (MAE)'.format(params['model_name']))
        plt.show()

        if return_stats:
            return X, Y, P, errors, error_stats
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    import os
    import yaml
    import tensorflow as tf
    import tools #local

    params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader )
    params['size_nan'] = int(params['len_input']*params['total_nan_share'])

    tools.set_gpu_configurations(params)
    model = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')

    run_test(model, params)
