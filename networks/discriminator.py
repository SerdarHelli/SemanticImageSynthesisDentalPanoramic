# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:21:17 2022

@author: pc
"""

from networks.layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def build_discriminator(image_shape, init_filters=16):
    input_image_A = keras.Input(shape=image_shape, name="discriminator_image_A")
    input_image_B = keras.Input(shape=image_shape, name="discriminator_image_B")
    x = layers.Concatenate()([input_image_A, input_image_B])
    x1 = downsample(4 *init_filters, 4, apply_norm=False)(x)
    x2 = downsample(8 * init_filters, 4)(x1)
    x3 = downsample(16 * init_filters, 4)(x2)
    x4 = downsample(32 * init_filters, 4, strides=1)(x3)
    x5 = layers.Conv2D(1, 4)(x4)
    outputs = [x1, x2, x3, x4, x5]
    return keras.Model([input_image_A, input_image_B], outputs)



def build_OASISdiscriminator(image_shape, init_filters=16,NUM_CLASSES=5):
    input = keras.Input(shape=image_shape, name="discriminator_image")
    x = layers.Conv2D(init_filters, 3, padding="same")(input)
    x1 = DownSampleBlock(init_filters)(x)
    x2 = DownSampleBlock(init_filters*2)(x1)
    x3 = DownSampleBlock(init_filters*4)(x2)
    x4 = DownSampleBlock(init_filters*8)(x3)
    x5 = DownSampleBlock(init_filters*16)(x4)

 
 
    y1 = UpSampleBlock(init_filters*8)(x5)
    y1= layers.concatenate([y1,x4])

    y2 = UpSampleBlock(init_filters*4)(y1)
    y2= layers.concatenate([y2,x3])

    y3 = UpSampleBlock(init_filters*2)(y2)
    y3= layers.concatenate([y3,x2])

    y4 = UpSampleBlock(init_filters)(y3)
    y4= layers.concatenate([y4,x1])


    y5 = UpSampleBlock(init_filters)(y4)

    outputs=layers.Conv2D(NUM_CLASSES+1, 3, padding="same",name="discriminator_output")(y5)
    return keras.Model(input, outputs)
