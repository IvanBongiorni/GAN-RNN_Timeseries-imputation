"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-19

Models implementation.
"""
import tensorflow as tf


def build_vanilla_seq2seq(params):
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
        Concatenate, TimeDistributed, Dense
    )

    ## ENCODER
    encoder_input = Input((params['len_input'], 17))

    # LSTM block
    encoder_lstm = LSTM(units = params['encoder_lstm_units'])(encoder_input)
    output_lstm = RepeatVector(params['len_input'])(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(encoder_input)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization()(conv_1)
    conv_2 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization()(conv_2)
    conv_3 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(conv_2)
    if params['use_batchnorm']:
        conv_3 = BatchNormalization()(conv_3)
    conv_4 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(conv_3)
    if params['use_batchnorm']:
        conv_4 = BatchNormalization()(conv_4)


    # Concatenate LSTM and Conv Encoder outputs for Decoder LSTM layer
    encoder_output = Concatenate(axis = -1)([output_lstm, conv_2])

    decoder_lstm = LSTM(params['decoder_dense_units'], return_sequences = True)(encoder_output)

    decoder_output = TimeDistributed(
        Dense(units = 1,
              activation = params['decoder_output_activation'],
              kernel_initializer = params['decoder_dense_initializer']))(decoder_lstm)

    seq2seq = Model(inputs = [encoder_input], outputs = [decoder_output])

    return seq2seq


def build_discriminator(params):
    '''
    Discriminator is based on the Vanilla seq2seq Encoder. The Decoder is removed
    and a Dense layer is left instead to perform binary classification.
    '''
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, Flatten, TimeDistributed, Dense
    )

    ## ENCODER
    encoder_input = Input((None, 17))

    # LSTM block
    encoder_lstm = LSTM(units = params['encoder_lstm_units'])(encoder_input)
    output_lstm = RepeatVector(params['len_input'])(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(encoder_input)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization()(conv_1)
    conv_2 = Conv1D(
        filters = params['conv_filters'],
        kernel_size = params['kernel_size'],
        activation = params['conv_activation'],
        kernel_initializer = params['conv_initializer'],
        padding = 'same')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization()(conv_2)

    # Concatenate LSTM and Conv Encoder outputs and Flatten for Decoder LSTM layer
    encoder_output = Concatenate(axis = -1)([output_lstm, conv_2])
    encoder_output = Flatten()(encoder_output)

    # Final layer for binary classification (real/fake)
    discriminator_output = Dense(
        units = 1,
        activation = 'sigmoid',
        kernel_initializer = params['decoder_dense_initializer'])(encoder_output)

    Discriminator = Model(inputs = [encoder_input], outputs = [discriminator_output])

    return Discriminator


def build_GAN(params):
    '''
    This is just a wrapper in case the model is trained as a GAN. It calls the vanilla
    seq2seq Generator, and build_discriminator() for the Discriminator model.
    '''
    generator = build_vanilla_seq2seq(params)
    discriminator = build_discriminator(params)
    return generator, discriminator
