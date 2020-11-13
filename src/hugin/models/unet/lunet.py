from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, ZeroPadding2D, Cropping2D, Convolution2D, Convolution2DTranspose, BatchNormalization, Bidirectional
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed, Layer
from tensorflow.keras.models import Model



class LSTMLastState(Layer):
    def __init__(self):
        super(LSTMLastState, self).__init__()

    def call(self, inputs):
        # print("LSTMLastState __call__ input: ", inputs)
        # (None, 5, 256, 256, 32)

        result = inputs[:, -1, :, :, :]
        # print("LSTMLastState __call__ output: ", result)
        return result


def encode_block(size, inputs, kernel, stride, activation, kinit, padding, max_pool=True, return_last_output=True, mask=None):
    result = []
    conv1_output, state1_h, state1_c = ConvLSTM2D(size, kernel_size=kernel, strides=stride, activation=activation,
                                                  kernel_initializer=kinit,
                                                  padding=padding, return_sequences=True, return_state=True, recurrent_dropout=0.2)(inputs)

    conv2_output, state2_h, state2_c = ConvLSTM2D(size, kernel_size=kernel, strides=stride, activation=activation,
                                                  kernel_initializer=kinit,
                                                  padding=padding, return_sequences=True, return_state=True)(
        conv1_output)

    result.append(conv2_output)
    if return_last_output:
        result.append(LSTMLastState()(conv2_output))

    result.extend([state2_h, state2_c])

    if max_pool:
        pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2_output)
        result.append(pool1)

    return result


def conv_t_block(size, input_1, input_2, kernel, stride, activation, kinit, padding, axis):
    conv1 = Convolution2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(input_1)
    if input_2 is not None:
        conv1 = concatenate([conv1, input_2], axis=axis)

    conv2 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv1)
    conv3 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv2)
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
        batch_norm=False,
        padding='same',
        axis=3,
        crop=0,
        mpadd=0,
):
    nr_classes = output_channels
    timeseries, input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]
    #timeseries_mask_shape = input_shapes["input_2"]
    inputs = Input((timeseries, input_1_height, input_1_width, input_1_channels))
    #masks = Input(timeseries_mask_shape)

    # Encoding
    conv1_output, conv1_output_last, state1_h, state1_c, pool1 = encode_block(32, inputs, kernel, stride, activation,
                                                                              kinit, padding)
    conv2_output, conv2_output_last, state2_h, state2_c, pool2 = encode_block(64, pool1, kernel, stride, activation,
                                                                              kinit, padding)
    conv3_output, conv3_output_last, state3_h, state3_c, pool3 = encode_block(128, pool2, kernel, stride, activation,
                                                                              kinit, padding)
    conv4_output, conv4_output_last, state4_h, state4_c, pool4 = encode_block(256, pool3, kernel, stride, activation,
                                                                              kinit, padding)

    # Middle
    conv5_output, conv5_output_last, state5_h, state5_c = encode_block(512, pool4, kernel, stride, activation, kinit,
                                                                       padding, max_pool=False)

    # Decoding
    conv6 = conv_t_block(256, conv5_output_last, conv4_output_last, kernel, stride, activation, kinit, padding, axis)
    conv7 = conv_t_block(128, conv6, conv3_output_last, kernel, stride, activation, kinit, padding, axis)
    conv8 = conv_t_block(64, conv7, conv2_output_last, kernel, stride, activation, kinit, padding, axis)
    conv9 = conv_t_block(32, conv8, conv1_output_last, kernel, stride, activation, kinit, padding, axis)

    # Output
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9

    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Convolution2D(nr_classes, (1, 1), activation='softmax', name="output_1")(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model