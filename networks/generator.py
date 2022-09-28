# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:17:12 2022

@author: pc
"""

from networks.layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_OASISgenerator(mask_shape, latent_dim=32):

    latent = keras.Input(shape=(mask_shape[0],mask_shape[1],latent_dim))
    mask = keras.Input(shape=mask_shape)


    y = NOISE3D(1024)(latent,mask)
    
    x = ResBlock(filters=1024)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=1024)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=1024)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=512)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=256)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=128)(y, mask)
    y = layers.UpSampling2D((2, 2))(x)

    x = tf.nn.leaky_relu(y, 0.2)
    output_image = tf.nn.tanh(layers.Conv2D(3, 3, padding="same")(x))
    
    return keras.Model([latent, mask], output_image, name="generator")





def build_generator(mask_shape, latent_dim=256):

    latent = keras.Input(shape=(latent_dim))
    mask = keras.Input(shape=mask_shape)
    x = layers.Dense(16384*2)(latent)
    x = layers.Reshape((8, 4, 1024))(x)
    
    x = ResBlock(filters=1024)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=1024)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=1024)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=512)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=256)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = ResBlock(filters=128)(x, mask)
    x = layers.UpSampling2D((2, 2))(x)

    x = tf.nn.leaky_relu(x, 0.2)
    output_image = tf.nn.tanh(layers.Conv2D(3, 4, padding="same")(x))
    
    return keras.Model([latent, mask], output_image, name="generator")



