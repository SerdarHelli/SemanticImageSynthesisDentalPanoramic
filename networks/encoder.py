# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:30:22 2022

@author: pc
"""

from networks.layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def build_encoder(image_shape, encoder_downsample_factor=16, latent_dim=256):
    input_image = keras.Input(shape=image_shape)
    x = downsample(4*encoder_downsample_factor, 3, apply_norm=False)(input_image)
    x = downsample(8 * encoder_downsample_factor, 3)(x)
    x = downsample(16 * encoder_downsample_factor, 3)(x)
    x = downsample(32 * encoder_downsample_factor, 3)(x)
    x = downsample(32 * encoder_downsample_factor, 3)(x)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dim, name="mean")(x)
    variance = layers.Dense(latent_dim, name="variance")(x)
    return keras.Model(input_image, [mean, variance], name="encoder")