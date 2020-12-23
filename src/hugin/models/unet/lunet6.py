from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, Cropping2D, Convolution2D, \
    Convolution2DTranspose, BatchNormalization, MaxPooling3D, Activation, LeakyReLU, PReLU
from tensorflow.keras.models import Model


def encode_block_lstm(size, inputs, kernel, stride, activation, kinit, padding, max_pool=True,
                      batch_normalization=False, mask=None):
    result = []
    use_bias = not batch_normalization
    x, state_h, state_c = ConvLSTM2D(size, kernel_size=kernel, strides=stride,
                                     kernel_initializer=kinit, use_bias=use_bias, activation='linear',
                                     padding=padding, return_sequences=True, return_state=True)(inputs, mask=mask)
    x = BatchNormalization()(x) if batch_normalization else x
    x = PReLU()(x) # In theory this should avoid the vanishing gradient situation that is, arguably more accute with RNNs

    x, state_h, state_c = ConvLSTM2D(size, kernel_size=kernel, strides=stride,
                                     kernel_initializer=kinit, use_bias=use_bias, activation='linear',
                                     padding=padding, return_sequences=True, return_state=True)(x,
                                                                                                mask=mask)  # can't set initial_state=(state_h, state_c) due to a bug in keras

    x = BatchNormalization()(x) if batch_normalization else x
    x = PReLU()(x)
    # result.append(x)
    result.append(state_c)

    if max_pool:
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(x)
        result.append(pool1)
    else:
        result.append(None)

    return result


def encode_block(size, inputs, kernel, stride, activation, kinit, padding, batch_normalization=False, max_pool=True):
    use_bias = not batch_normalization

    conv1 = Convolution2D(size, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(inputs)
    conv1 = BatchNormalization()(conv1) if batch_normalization else conv1
    conv1 = Activation(activation)(conv1)

    conv1 = Convolution2D(size, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_normalization else conv1
    conv1 = Activation(activation)(conv1)


    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) if max_pool else None

    return conv1, pool1


def conv_t_block(size, input_1, input_2, kernel, stride, activation, kinit, padding, axis, batch_normalization=False):
    use_bias = not batch_normalization

    conv1 = Convolution2DTranspose(256, (2, 2), strides=(2, 2), padding=padding, use_bias=use_bias)(input_1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    if input_2 is not None:
        input_2 = BatchNormalization()(input_2)
        conv1 = concatenate([conv1, input_2], axis=axis)

    conv2 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv1)
    conv2 = BatchNormalization()(conv2) if batch_normalization else conv2
    conv2 = Activation(activation)(conv2)

    conv3 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3) if batch_normalization else conv2
    conv3 = Activation(activation)(conv3)

    return conv3


def unet_rrn(
        name,
        input_shapes,
        output_shapes,
        kernel=3,
        stride=1,
        activation='elu',
        output_channels=2,
        kinit='RandomUniform',
        batch_norm=True,
        padding='same',
        axis=3,
        crop=0,
        mpadd=0,
):
    nr_classes = output_channels
    timeseries, input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]
    timeseries_mask_shape = input_shapes["input_2"]
    inputs = Input((timeseries, input_1_height, input_1_width, input_1_channels))
    mask = Input(timeseries_mask_shape)

    # Encoding
    conv1_output_last, pool1 = encode_block_lstm(32, inputs, kernel, stride, activation, kinit, padding, mask=mask,
                                                 batch_normalization=True)
    conv2_output_last, pool2 = encode_block_lstm(64, pool1, kernel, stride, activation, kinit, padding,
                                                 batch_normalization=True)
    conv3_output_last, _ = encode_block_lstm(128, pool2, kernel, stride, activation, kinit, padding, max_pool=False,
                                             batch_normalization=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_output_last)
    conv4_output_last, pool4 = encode_block(256, pool3, kernel, stride, activation, kinit, padding,
                                            batch_normalization=batch_norm)

    # Middle
    conv5_output_last, _ = encode_block(512, pool4, kernel, stride, activation, kinit, padding, max_pool=False,
                                        batch_normalization=batch_norm)

    # Decoding
    conv6 = conv_t_block(256, conv5_output_last, conv4_output_last, kernel, stride, activation, kinit, padding, axis,
                         batch_normalization=batch_norm)
    conv7 = conv_t_block(128, conv6, conv3_output_last, kernel, stride, activation, kinit, padding, axis,
                         batch_normalization=batch_norm)
    conv8 = conv_t_block(64, conv7, conv2_output_last, kernel, stride, activation, kinit, padding, axis,
                         batch_normalization=batch_norm)
    conv9 = conv_t_block(32, conv8, conv1_output_last, kernel, stride, activation, kinit, padding, axis,
                         batch_normalization=batch_norm)

    # Output

    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Convolution2D(nr_classes, (1, 1), activation='softmax', name="output_1")(conv9)
    model = Model(inputs=[inputs, mask], outputs=[conv10])

    # model.summary()
    # import sys
    # sys.exit(1)
    return model
