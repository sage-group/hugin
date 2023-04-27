# -*- coding: utf-8 -*-
__license__ = """Copyright 2023 West University of Timisoara

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

import tensorflow as tf



from .base import ModelBuilder
from ...tools.utils import dice_coef


class UNet(ModelBuilder):
    def __init__(
        self,
        *args,
        freeze_encoding: bool = False,
        freeze_decoding: bool = False,
        dropout: float = 0.25,
        kernel: float = 3,
        stride: int = 1,
        batch_normalization: float = None,
        axis: int = 3,
        padding: str = "same",
        kinit: str = "RandomUniform",
        activation: str = "PReLU",
        mpad: int = 0,
        custom_objects: dict = {},
        **kwargs,
    ):
        custom_objects["dice_coef"] = dice_coef
        super().__init__(*args, custom_objects=custom_objects, **kwargs)
        self.freeze_encoding = freeze_encoding
        self.freeze_decoding = freeze_decoding
        self.dropout = dropout
        self.kernel = kernel
        self.stride = stride
        self.batch_normalization = batch_normalization
        self.axis = axis
        self.padding = padding
        self.kinit = kinit
        self.activation = activation  # 'PReLU'
        self.mpad = mpad

    def __call__(self, input_shapes,
                 output_shapes,
                 output_channels,
                 name=None, crop=0):
        input_1_height, input_1_width, input_1_channels = input_shapes["input_1"]
        inputs = tf.keras.layers.Input((input_1_height,
                                        input_1_width,
                                        input_1_channels))

        # Encoding
        conv1_output_last, pool1 = self.encode_block_lstm(
            32,
            inputs,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            trainable=not self.freeze_encoding,
        )
        conv2_output_last, pool2 = self.encode_block_lstm(
            64,
            pool1,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            trainable=not self.freeze_encoding,
        )
        conv3_output_last, pool3 = self.encode_block_lstm(
            128,
            pool2,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            trainable=not self.freeze_encoding,
        )
        conv4_output_last, pool4 = self.encode_block_lstm(
            256,
            pool3,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            trainable=not self.freeze_encoding,
        )

        # Middle
        conv5_output_last, _ = self.encode_block_lstm(
            512,
            pool4,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            max_pool=False,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            trainable=not self.freeze_encoding,
        )

        # Decoding
        conv6 = self.conv_t_block(
            256,
            conv5_output_last,
            conv4_output_last,
            self.kernel,
            self.stride,
            activation=self.activation,
            kinit=self.kinit,
            padding=self.padding,
            concatenate_axis=self.axis,
            trainable=not self.freeze_decoding,
        )
        conv7 = self.conv_t_block(
            128,
            conv6,
            conv3_output_last,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            concatenate_axis=self.axis,
            trainable=not self.freeze_decoding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
        )
        conv8 = self.conv_t_block(
            64,
            conv7,
            conv2_output_last,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            concatenate_axis=self.axis,
            trainable=not self.freeze_decoding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
        )
        conv9 = self.conv_t_block(
            32,
            conv8,
            conv1_output_last,
            self.kernel,
            self.stride,
            self.activation,
            self.kinit,
            padding=self.padding,
            concatenate_axis=self.axis,
            trainable=not self.freeze_decoding,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
        )

        # Output

        conv9 = tf.keras.layers.Cropping2D((self.mpad, self.mpad))(conv9)

        conv10 = tf.keras.layers.Convolution2D(
            output_channels, (1, 1), activation="softmax", name="output_1"
        )(conv9)
        model = tf.keras.Model(
            inputs=[
                inputs,
            ],
            outputs=[conv10],
        )
        return model

    def encode_block_lstm(
        self,
        size,
        inputs,
        kernel,
        stride,
        activation,
        kinit,
        padding,
        max_pool=True,
        batch_normalization=None,
        dropout=None,
        trainable=True,
    ):
        result = []
        use_bias = not batch_normalization
        x = tf.keras.layers.Convolution2D(
            size,
            kernel_size=kernel,
            strides=stride,
            kernel_initializer=kinit,
            use_bias=use_bias,
            activation="linear",
            padding=padding,
            trainable=trainable,
        )(inputs)
        x = (
            tf.keras.layers.BatchNormalization(trainable=trainable)(x)
            if batch_normalization
            else x
        )
        x = tf.keras.layers.Activation(activation, trainable=trainable)(x)

        x = tf.keras.layers.Convolution2D(
            size,
            kernel_size=kernel,
            strides=stride,
            kernel_initializer=kinit,
            use_bias=use_bias,
            activation="linear",
            padding=padding,
            trainable=trainable,
        )(x)

        x = (
            tf.keras.layers.BatchNormalization(trainable=trainable)(x)
            if batch_normalization
            else x
        )

        result.append(x)

        if max_pool:
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 trainable=trainable)(x)
            pool1 = (
                tf.keras.layers.Dropout(dropout, trainable=trainable)(pool1)
                if dropout
                else pool1
            )
            result.append(pool1)
        else:
            result.append(None)

        return result

    def conv_t_block(
        self,
        size,
        input_1,
        input_2,
        kernel,
        stride,
        activation,
        kinit,
        padding,
        concatenate_axis=3,
        batch_normalization=None,
        dropout=None,
        trainable=True,
    ):
        use_bias = not batch_normalization

        conv1 = tf.keras.layers.Convolution2DTranspose(
            size,
            (2, 2),
            strides=(2, 2),
            padding=padding,
            use_bias=use_bias,
            trainable=trainable,
        )(input_1)
        conv1 = (
            tf.keras.layers.BatchNormalization(trainable=trainable)(conv1)
            if batch_normalization
            else conv1
        )
        conv1 = tf.keras.layers.Activation(activation, trainable=trainable)(conv1)

        if input_2 is not None:
            input_2 = (
                tf.keras.layers.BatchNormalization(trainable=trainable)(input_2)
                if batch_normalization
                else input_2
            )
            conv1 = tf.keras.layers.concatenate(
                [conv1, input_2], axis=concatenate_axis, trainable=trainable
            )
            conv1 = (
                tf.keras.layers.Dropout(dropout, trainable=trainable)(conv1)
                if dropout
                else conv1
            )

        conv2 = tf.keras.layers.Convolution2D(
            size,
            kernel_size=kernel,
            strides=stride,
            kernel_initializer=kinit,
            padding=padding,
            use_bias=use_bias,
            trainable=trainable,
        )(conv1)
        conv2 = (
            tf.keras.layers.BatchNormalization(trainable=trainable)(conv2)
            if batch_normalization
            else conv2
        )
        conv2 = tf.keras.layers.Activation(activation, trainable=trainable)(conv2)

        conv3 = tf.keras.layers.Convolution2D(
            size,
            kernel_size=kernel,
            strides=stride,
            kernel_initializer=kinit,
            padding=padding,
            use_bias=False,
            trainable=trainable,
        )(conv2)
        conv3 = (
            tf.keras.layers.BatchNormalization(trainable=trainable)(conv3)
            if batch_normalization
            else conv2
        )
        conv3 = tf.keras.layers.Activation(activation, trainable=trainable)(conv3)

        return conv3
