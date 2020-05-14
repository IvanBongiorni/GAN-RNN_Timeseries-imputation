"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-19

Model implementation.
"""
import tensorflow as tf


def build(params):
    """
    Implements a seq2seq RNN with Convolutional self attention. It keeps a canonical
    Encoder-Decoder structure: an Embedding layers receives the sequence of chars and
    learns a representation. This series is received by two different layers at the same time.
    First, an LSTM Encoder layer, whose output is repeated and sent to the Decoder. Second, a
    block of 1D Conv layers. Their kernel filters work as multi-head self attention layers.
    All their scores are pushed through a TanH gate that scales each score in the [-1,1] range.
    Both LSTM and Conv outputs are concatenated and sent to an LSTM Decoder, that processes
    the signal and sents it to Dense layers, performing the prediction for each step of the
    output series.

    Args: params dict
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, LSTM, TimeDistributed, Dense
    )

    ## ENCODER
    encoder_input = Input((params['len_input'], 1))

    # LSTM block
    encoder_lstm = LSTM(units = params['len_input'])(encoder_input)
    output_lstm = RepeatVector(params['len_input'])(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(filters = params['conv_filters'][0],
                    kernel_size = params['kernel_size'],
                    activation = params['conv_activation'],
                    kernel_initializer = params['conv_initializer'],
                    padding = 'same')(encoder_input)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv1D(filters = params['conv_filters'][1],
                    kernel_size = params['kernel_size'],
                    activation = params['conv_activation'],
                    kernel_initializer = params['conv_initializer'],
                    padding = 'same')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization()(conv_2)

    concatenation = Concatenate(axis = -1)([output_lstm, conv_2])

    ## DECODER
    decoder_lstm = LSTM(params['len_input'], return_sequences = True)(concatenation)
    decoder_dense = TimeDistributed(
        Dense(params['decoder_dense_units'],
              activation = params['decoder_dense_activation'],
              kernel_initializer = params['decoder_dense_initializer'])
        )(decoder_lstm)
    decoder_output = TimeDistributed(
        Dense(units = 1,
              activation = params['decoder_output_activation'],
              kernel_initializer = params['decoder_dense_initializer'])
        )(decoder_dense)


    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    return model



################################################################################################
###    GAN TRAINING IS STILL A WORK IN PROGRESS - DO NOT TOUCH UNTIL VANILLA TRAINING IS READY
################################################################################################


def build_discriminator(params):
    """
    Discriminator is based on the Vanilla seq2seq architecture. They only differ by
    the last two layers, since this is a classifier and not a regressor.
    Models are made as symmetric as possible to allow for balanced adversarial training.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, LSTM, TimeDistributed, Dense
    )

    ## ENCODER
    encoder_input = Input((params['len_input'], 1))

    # LSTM block
    encoder_lstm = LSTM(params['len_input'])(encoder_input)
    output_lstm = RepeatVector(params['len_input'])(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(filters = params['conv_filters'][0], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
                    padding = 'same')(encoder_input)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv1D(filters = params['conv_filters'][1], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
                    padding = 'same')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization()(conv_2)

    concatenation = Concatenate(axis = -1)([output_lstm, conv_2])

    ## DECODER
    decoder_lstm = LSTM(params['len_input'])(concatenation)
    decoder_dense = Dense(params['decoder_dense_units'], activation = params['decoder_dense_activation'])(decoder_lstm)
    decoder_output = Dense(1, activation = 'sigmoid')(decoder_dense)


    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    return model


def build_GAN(params):
    '''
    This is just a wrapper in case the model is trained as a GAN. It calls the vanilla
    seq2seq Generator, and build_discriminator() for the Discriminator model.
    '''
    generator = build(params)
    discriminator = build_discriminator(params)
    return generator, discriminator
