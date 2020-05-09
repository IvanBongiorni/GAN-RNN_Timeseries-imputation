"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-19

Model implementation.
"""
import tensorflow as tf
# Prevents CuDNN 'Failed to get convolution algorithm' error
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices, True)

# # Solves Convolution CuDNN error
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


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

    # ENCODER
    encoder_input = Input((params['len_input'], 1))

    # LSTM block
    encoder_lstm = LSTM(params['encoder_lstm_units'])(encoder_input)
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

    # DECODER
    decoder_lstm = LSTM(params['len_input'], return_sequences = True)(concatenation)
    decoder_dense = TimeDistributed(Dense(params['decoder_dense_units'], activation = params['decoder_dense_activation']))(decoder_lstm)
    decoder_output = TimeDistributed(Dense(1, activation = params['decoder_output_activation']))(decoder_dense)


    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    return model


# def build_discriminator(params):
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import (
#         Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
#         Concatenate, LSTM, TimeDistributed, Dense
#     )
#
#     # ENCODER
#     encoder_input = Input((params['len_input']), name = 'input')
#     encoder_lstm = LSTM(params['encoder_lstm_units'], name = 'encoder_lstm')(encoder_input)
#     encoder_output = RepeatVector(params['decoder_lstm_units'], name = 'encoder_output')(encoder_lstm)
#
#     # Conv block
#     conv_1 = Conv1D(filters = params['conv_filters'][0], kernel_size = params['kernel_size'],
#                     activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
#                     padding = 'same', name = 'conv1')(encoder_input)
#     if params['use_batchnorm']:
#         conv_1 = BatchNormalization(name = 'batchnorm_1')(conv_1)
#
#     conv_2 = Conv1D(filters = params['conv_filters'][1], kernel_size = params['kernel_size'],
#                     activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
#                     padding = 'same', name = 'conv2')(conv_1)
#     if params['use_batchnorm']:
#         conv_2 = BatchNormalization(name = 'batcnorm_2')(conv_2)
#
#     attention = tf.nn.tanh(conv_2, name = 'attention')
#
#     # DECODER
#     concatenation = Concatenate(axis = -1, name = 'concatenation')([encoder_output, attention])
#
#     decoder_lstm = LSTM(params['decoder_lstm_units'], name = 'decoder_lstm')(concatenation)
#
#     decoder_dense = Dense(params['discriminator_dense_units'], activation = params['decoder_dense_activation'],
#                           name = 'discriminator_decoder_dense')(decoder_lstm)
#     decoder_output = Dense(1, activation = 'sigmoid', name = 'decoder_output')(decoder_dense)
#
#     discriminator = Model(inputs = [encoder_input], outputs = [decoder_output])
#     return discriminator
#
#
# def build_GAN(params):
#     '''
#     This is just a wrapper in case the model is trained as a GAN. In this case,
#     it calls the vanilla seq2seq as Generator, and calls build_discriminator() for
#     the Discriminator model, that follows the same structure, except for the last
#     two layers - ends with one node + Sigmoid activation for binary classification
#     '''
#     generator = build(params)
#     discriminator = build_discriminator(params)
#     return generator, discriminator
