# -*- coding: utf-8 -*-
__license__ = \
    """Copyright 2019 West University of Timisoara
    
       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at
    
           http://www.apache.org/licenses/LICENSE-2.0
    
       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, ZeroPadding2D, Cropping2D, Convolution2D, Convolution2DTranspose, BatchNormalization, Bidirectional, Dropout, Activation, SpatialDropout2D
from tensorflow.keras.models import Model

from hugin.tools.utils import custom_objects, dice_coef, MultilabelMeanIOU


@custom_objects({'dice_coef': dice_coef})
def unet_v15(
        name,
        input_shapes,
        output_shapes,
        kernel=3,
        stride=1,
        activation='elu',
        output_channels=2,
        kinit='RandomUniform',
        batch_norm=True,
        dropout=True,
        dropout_rate=0.1,
        padding='same',
        axis=3,
        crop=0,
        mpadd=0,
             ):
    nr_classes = output_channels
    use_bias = not batch_norm
    if len(input_shapes["input_1"]) == 3:
        input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]
    else:
        timestamps_1, input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]


    inputs = Input((input_1_height, input_1_width, input_1_channels))

    conv1 = ZeroPadding2D((crop, crop))(inputs)

    ################

    conv1 = Convolution2D(32, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv1)

    conv1 = BatchNormalization()(conv1) if batch_norm else conv1
    conv1 = Activation(activation)(conv1)
    conv1 = SpatialDropout2D(dropout_rate)(conv1) if dropout else conv1

    conv1 = Convolution2D(32, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_norm else conv1
    conv1 = Activation(activation)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(dropout_rate)(pool1) if dropout else pool1

    ################

    conv2 = Convolution2D(64, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(pool1)
    conv2 = BatchNormalization()(conv2) if batch_norm else conv2
    conv2 = Activation(activation)(conv2)
    conv2 = SpatialDropout2D(dropout_rate)(conv2) if dropout else conv2

    conv2 = Convolution2D(64, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv2)
    conv2 = BatchNormalization()(conv2) if batch_norm else conv2
    conv2 = Activation(activation)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = SpatialDropout2D(dropout_rate)(pool2) if dropout else pool2


    ################

    conv3 = Convolution2D(128, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(pool2)
    conv3 = BatchNormalization()(conv3) if batch_norm else conv3
    conv3 = Activation(activation)(conv3)
    conv3 = SpatialDropout2D(dropout_rate)(conv3) if dropout else conv3

    conv3 = Convolution2D(128, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv3)
    conv3 = BatchNormalization()(conv3) if batch_norm else conv3
    conv3 = Activation(activation)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = SpatialDropout2D(dropout_rate)(pool3) if dropout else pool3

    ################

    conv4 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(pool3)
    conv4 = BatchNormalization()(conv4) if batch_norm else conv4
    conv4 = Activation(activation)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4) if dropout else conv4

    conv4 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv4)
    conv4 = BatchNormalization()(conv4) if batch_norm else conv4
    conv4 = Activation(activation)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = SpatialDropout2D(dropout_rate)(pool4) if dropout else pool4

    ################

    conv5 = Convolution2D(512, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(pool4)
    conv5 = BatchNormalization()(conv5) if batch_norm else conv5
    conv5 = Activation(activation)(conv5)
    conv5 = SpatialDropout2D(dropout_rate)(conv5) if dropout else conv5

    conv5 = Convolution2D(512, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv5 = SpatialDropout2D(dropout_rate)(conv5) if dropout else conv5

    up6 = concatenate([Convolution2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(conv5), conv4], axis=axis)

    conv6 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(up6)
    conv6 = BatchNormalization()(conv6) if batch_norm else conv6
    conv6 = Activation(activation)(conv6)
    conv6 = SpatialDropout2D(dropout_rate)(conv6) if dropout else conv6

    conv6 = Convolution2D(256, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv6)
    conv6 = BatchNormalization()(conv6) if batch_norm else conv6
    conv6 = Activation(activation)(conv6)

    up7 = concatenate([Convolution2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(conv6), conv3], axis=axis)

    conv7 = Convolution2D(128, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(up7)
    conv7 = BatchNormalization()(conv7) if batch_norm else conv7
    conv7 = Activation(activation)(conv7)
    conv7 = SpatialDropout2D(dropout_rate)(conv7) if dropout else conv6

    conv7 = Convolution2D(128, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv7)
    conv7 = BatchNormalization()(conv7) if batch_norm else conv7
    conv7 = Activation(activation)(conv7)

    up8 = concatenate([Convolution2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(conv7), conv2], axis=axis)

    conv8 = Convolution2D(64, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(up8)
    conv8 = BatchNormalization()(conv8) if batch_norm else conv8
    conv8 = Activation(activation)(conv8)
    conv8 = SpatialDropout2D(dropout_rate)(conv8) if dropout else conv8

    conv8 = Convolution2D(64, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv8)
    conv8 = BatchNormalization()(conv8) if batch_norm else conv8
    conv8 = Activation(activation)(conv8)

    up9 = concatenate([Convolution2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(conv8), conv1], axis=axis)
    conv9 = Convolution2D(32, kernel_size=kernel, strides=stride, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(up9)
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9
    conv9 = Activation(activation)(conv9)
    conv9 = SpatialDropout2D(dropout_rate)(conv9) if dropout else conv9
    conv9 = Convolution2D(32, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding, use_bias=use_bias)(conv9)
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9
    conv9 = Activation(activation)(conv9)

    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Convolution2D(nr_classes, (1, 1), activation='softmax', name="output_1")(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
