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

from keras.layers import Input, concatenate, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.layers.convolutional import Convolution2D, Convolution2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from hugin.tools.utils import custom_objects, dice_coef


@custom_objects({'dice_coef': dice_coef})
def unet_v14(
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
    input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]

    inputs = Input((input_1_height, input_1_width, input_1_channels))

    conv1 = ZeroPadding2D((crop, crop))(inputs)

    conv1 = Convolution2D(32, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv1)
    conv1 = Convolution2D(32, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(pool1)
    conv2 = Convolution2D(64, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(pool2)
    conv3 = Convolution2D(128, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(pool3)
    conv4 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(pool4)
    conv5 = Convolution2D(512, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv5)

    up6 = concatenate([Convolution2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(conv5), conv4], axis=axis)

    conv6 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(up6)
    conv6 = Convolution2D(256, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv6)

    up7 = concatenate([Convolution2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(conv6), conv3], axis=axis)

    conv7 = Convolution2D(128, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(up7)
    conv7 = Convolution2D(128, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv7)

    up8 = concatenate([Convolution2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(conv7), conv2], axis=axis)

    conv8 = Convolution2D(64, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(up8)
    conv8 = Convolution2D(64, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv8)

    up9 = concatenate([Convolution2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(conv8), conv1], axis=axis)
    conv9 = Convolution2D(32, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(up9)
    conv9 = Convolution2D(32, kernel_size=kernel, strides=stride, activation=activation, kernel_initializer=kinit,
                          padding=padding)(conv9)
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9

    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Convolution2D(nr_classes, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
