"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-24

Main script - Wrapper of the whole processing+training pipeline

Imports and processes data, loads config params, runs processing pipeline, builds 
model (either plain or GAN) and trains it, saves it in dedicated folder.
"""

def run():
    import os
    import yaml, pickle
    import time
    from pdb import set_trace as BP
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    # local modules
    import processing, model, train
    
    # Get current directory
    current_path = os.getcwd()
    
    # Load configuration params
    params = yaml.load(open(current_path + '/config.yaml'), yaml.Loader)
    
    # Update params with path data
    params['data_path'] = current_path + '/data/'
    params['save_path'] = current_path + '/saved_models/'
    
    # Load data
    X, V, Y = processing.load_and_process_data(params)
    
    # Load and train model
    if params['gan']:
        Generator, Adversary = model.build_GAN(params)
        train.train_GAN(Generator, Adversary, X, V, params)
    else:
        CRNN = model.build(params)
        train.train(CRNN, X, V, params)
    
    # Check performance on Test data
    chech_performance_on_test_data(Y)
    
    return None



if __name__ == '__main__':
    run()
