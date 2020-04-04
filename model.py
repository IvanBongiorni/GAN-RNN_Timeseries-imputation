"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-19

Model implementation.
"""


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
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, LSTM, TimeDistributed, Dense
    )

    # ENCODER
    encoder_input = Input((params['len_input']), name = 'input')

    encoder_lstm = LSTM(params['encoder_lstm_units'],
                        name = 'encoder_lstm')(encoder_input)

    encoder_output = RepeatVector(params['decoder_lstm_units'],
                                  name = 'encoder_output')(encoder_lstm)


    # Convolutional Self-Attention
    conv_1 = Conv1D(filters = params['conv_filters'][0], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
                    padding = 'same', name = 'conv1')(encoder_input)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization(name = 'batchnorm_1')(conv_1)

    conv_2 = Conv1D(filters = params['conv_filters'][1], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
                    padding = 'same', name = 'conv2')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization(name = 'batcnorm_2')(conv_2)

    # conv_3 = Conv1D(filters = params['conv_filters'][2], kernel_size = params['kernel_size'],
    #                 activation = params['conv_activation'], kernel_initializer = params['conv_initializer'],
    #                 padding = 'same', name = 'conv3')(conv_2)
    # if params['use_batchnorm']:
    #     conv_3 = BatchNormalization(name = 'batchnorm_3')(conv_3)

    attention = tf.nn.tanh(conv_2, name = 'attention')


    # DECODER
    concatenation = Concatenate(axis = -1, name = 'concatenation')([encoder_output, attention])

    decoder_lstm = LSTM(params['decoder_lstm_units'], return_sequences = True,
                        name = 'decoder_lstm')(concatenation)

    decoder_dense = TimeDistributed(Dense(params['decoder_dense_units'],
                                          activation = params['decoder_dense_activation']),
                                    name = 'decoder_dense')(decoder_lstm)

    decoder_output = TimeDistributed(Dense(1, activation = params['decoder_output_activation']),
                                     name = 'decoder_output')(decoder_dense)


    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    return model


def build_GAN():
    import tensorflow as tf
    from tensorflow.keras.models import Model

    G = build(params)

    A = Sequential([

        ## VALUTA L'IPOTESI DI INSERIRE LO STESSO INPUT IN LSTM e Conv, e poi riunirli in Dense

        LSTM(params['adversary_lstm_units'], input_shape)

        Dense(params['adversary_dense_1'], activation = params['adversary_activation']),

        Dense(params['adversary_dense_2'], activation = params['adversary_activation']),

        Dense(1, activation = 'sigmoid')

    ])

    return G, A
