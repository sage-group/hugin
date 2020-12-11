from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, ZeroPadding2D, Cropping2D, Convolution2D, \
    Convolution2DTranspose, BatchNormalization, Bidirectional
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed, Layer
from tensorflow.keras.models import Model

from hugin.tools.utils import custom_objects


class LSTMLastState(Layer):
    def __init__(self, *args, **kwargs):
        super(LSTMLastState, self).__init__(*args, **kwargs)

    def call(self, inputs):
        print("LSTMLastState __call__ input: ", inputs)
        # (None, 5, 256, 256, 32)

        result = inputs[:, -1, :, :, :]
        # print("LSTMLastState __call__ output: ", result)
        return result


def encode_block_lstm(size, inputs, kernel, stride, activation, kinit, padding, max_pool=True, batch_normalization=False, mask=None):
    result = []
    x = ConvLSTM2D(size, kernel_size=kernel, strides=stride, activation=activation,
                   kernel_initializer=kinit,
                   padding=padding, return_sequences=True)(inputs, mask=mask)

    x = ConvLSTM2D(size, kernel_size=kernel, strides=stride, activation=activation,
                   kernel_initializer=kinit,
                   padding=padding, return_sequences=False)(x, mask=mask)

    x = BatchNormalization()(x) if batch_normalization else x
    result.append(x)

    if max_pool:
        pool1 = MaxPooling2D(pool_size=(2, 2))(x)
        result.append(pool1)

    return result


def encode_block(size, inputs, kernel, stride, activation, kinit, padding, batch_normalization=False, max_pool=True):
    conv1 = Convolution2D(size, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(inputs)
    conv1 = BatchNormalization()(conv1) if batch_normalization else conv1
    conv1 = Convolution2D(size, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_normalization else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) if max_pool else None

    return conv1, pool1


def conv_t_block(size, input_1, input_2, kernel, stride, activation, kinit, padding, axis, batch_normalization=False):
    conv1 = Convolution2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(input_1)
    conv1 = BatchNormalization()(conv1)
    if input_2 is not None:
        conv1 = concatenate([conv1, input_2], axis=axis)

    conv2 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv1)
    conv2 = BatchNormalization()(conv2) if batch_normalization else conv2
    conv3 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv2)
    return conv3


@custom_objects({'LSTMLastState': LSTMLastState})
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
    conv1_output_last, pool1 = encode_block_lstm(32, inputs, kernel, stride, activation, kinit, padding, mask=mask)
    conv2_output_last, pool2 = encode_block(64, pool1, kernel, stride, activation, kinit, padding)
    conv3_output_last, pool3 = encode_block(128, pool2, kernel, stride, activation, kinit, padding)
    conv4_output_last, pool4 = encode_block(256, pool3, kernel, stride, activation, kinit, padding)

    # Middle
    conv5_output_last, _ = encode_block(512, pool4, kernel, stride, activation, kinit, padding, max_pool=False)

    # Decoding
    conv6 = conv_t_block(256, conv5_output_last, conv4_output_last, kernel, stride, activation, kinit, padding, axis)
    conv7 = conv_t_block(128, conv6, conv3_output_last, kernel, stride, activation, kinit, padding, axis)
    conv8 = conv_t_block(64, conv7, conv2_output_last, kernel, stride, activation, kinit, padding, axis)
    conv9 = conv_t_block(32, conv8, conv1_output_last, kernel, stride, activation, kinit, padding, axis)

    # Output
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9

    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Convolution2D(nr_classes, (1, 1), activation='softmax', name="output_1")(conv9)
    model = Model(inputs=[inputs, mask], outputs=[conv10])

    return model
