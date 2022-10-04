# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:10:05 2022

@author: pc
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_addons as tfa

class NOISE3D(layers.Layer):
    def __init__(self,filters,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.targeted_shape=4
        self.patch_size=4


    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, 3, padding="same")
        self.spade= SPADE(self.filters)

    def call(self,latent,masks):
        x=tf.concat([latent,masks],axis=-1)
        x=tf.image.resize(x, (self.targeted_shape,self.targeted_shape), method="bilinear")
        x=self.conv1(x)
        return x



class GaussianSampler(layers.Layer):
    def __init__(self, batch_size, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def call(self, inputs):
        means, variance = inputs
        epsilon = tf.random.normal(
            shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=1.0
        )
        samples = means + tf.exp(0.5 * variance) * epsilon
        return samples


class SPADE(layers.Layer):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
    
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.filters=filters
        self.conv = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_gamma = layers.Conv2D(filters, 3, padding="same")
        self.conv_beta = layers.Conv2D(filters, 3, padding="same")

    def build(self, input_shape):

        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, (self.resize_shape), method="nearest")
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = ((gamma) * normalized) + beta
        return output






class ResBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.spade_1 = SPADE(input_filter)
        self.spade_2 = SPADE(self.filters)
        self.conv_1 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_2 = layers.Conv2D(self.filters, 3, padding="same")
        self.learned_skip = False
        

        if self.filters != input_filter:
            self.learned_skip = True
            self.spade_3 = SPADE(input_filter)
            self.conv_3 = layers.Conv2D(self.filters, 3, padding="same")

    def call(self, input_tensor, mask):
        x = self.spade_1(input_tensor, mask)
        x = self.conv_1(tf.nn.leaky_relu(x, 0.2))
        x = self.spade_2(x, mask)
        x = self.conv_2(tf.nn.leaky_relu(x, 0.2))
        skip = (
            self.conv_3(tf.nn.leaky_relu(self.spade_3(input_tensor, mask), 0.2))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x

        return output



class DownSampleBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_2 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_3 = layers.Conv2D(self.filters, 3, padding="same")
        self.downsample=layers.AveragePooling2D(pool_size=(2, 2))
        self.norm1=tfa.layers.InstanceNormalization()
        self.norm2=tfa.layers.InstanceNormalization()
        self.norm3=tfa.layers.InstanceNormalization()

    def call(self, input_tensor):
        x =self.norm1(self.conv_1(tf.nn.leaky_relu(input_tensor, 0.2)))
        x=self.downsample(tf.nn.leaky_relu(x))
        x = self.norm2(self.conv_2(x))
        x = self.norm3(self.conv_3(tf.nn.leaky_relu(x, 0.2)))
        return x


class UpSampleBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_2 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_3 = layers.Conv2D(self.filters, 3, padding="same")
        self.upsample=layers.UpSampling2D((2, 2))
        self.norm1=tfa.layers.InstanceNormalization()
        self.norm2=tfa.layers.InstanceNormalization()
        self.norm3=tfa.layers.InstanceNormalization()

    def call(self, input_tensor):
        x = self.norm1(self.conv_1(tf.nn.leaky_relu(input_tensor, 0.2)))
        x=self.upsample(tf.nn.leaky_relu(x,0.2))
        x = self.norm2(self.conv_2(x))
        x = self.norm3(self.conv_3(tf.nn.leaky_relu(x, 0.2)))
        return x

def downsample(
    channels,
    kernels,
    strides=2,
    apply_norm=True,
    apply_activation=True,
    apply_dropout=False,
):
    block = keras.Sequential()
    block.add(
        layers.Conv2D(
            channels,
            kernels,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.GlorotNormal(),
        )
    )
    if apply_norm:
        block.add(tfa.layers.InstanceNormalization())
    if apply_activation:
        block.add(layers.LeakyReLU(0.2))
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    return block



