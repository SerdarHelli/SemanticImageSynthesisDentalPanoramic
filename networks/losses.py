# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:11:54 2022

@author: pc
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np

def kl_divergence_loss(mean, variance):
    return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))

def generator_loss(y):
    return -tf.reduce_mean(y)

class FeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true) - 1):
            loss += self.mae(y_true[i], y_pred[i])
        return loss

class StructureSmilarity(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_val=255
        self.filter_size=11
        self.filter_sigma=1.5
        self.k1=0.01
        self.k2=0.03

    def get_ssim(self, y_true, y_pred):
         return tf.image.ssim(y_true, y_pred ,max_val=self.max_val, filter_size=self.filter_size,
                                  filter_sigma=self.filter_sigma, k1=self.k1, k2=self.k2)

    def call(self, y_true, y_pred):
      y_true=tf.cast(((y_true*127.5)+127.5),tf.uint8)
      y_pred=tf.cast(((y_pred*127.5)+127.5),tf.uint8)
      loss=self.get_ssim(y_true, y_pred)
      return tf.reduce_mean(loss)

      
class VGGFeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = keras.Model(vgg.input, layer_outputs, name="VGG")
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = keras.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = keras.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss


class ThresholdedFeatureLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mae = keras.losses.MeanAbsoluteError()


    def make_threshold(self,labels,image):
  
        labels=labels[:,:,:,0] +labels[:,:,:,1] +labels[:,:,:,2] 
        boolenmask=tf.math.greater_equal(labels,  tf.constant(-2, dtype=tf.float32))
        shape = tf.shape(boolenmask)
        boolenmask=tf.reshape(boolenmask,(shape[0],shape[1],shape[2],1))
        boolenmask=tf.concat((boolenmask,boolenmask,boolenmask),axis=3)
        thresholded_img=tf.where(boolenmask,image, tf.constant(0, dtype=tf.float32))
        return thresholded_img

    def call(self, y_true, y_pred):

        y,labels=y_true[0],y_true[1]
        thresholded_true=self.make_threshold(y,labels)
        thresholded_pred=self.make_threshold(y_pred,labels)

        loss=self.mae(thresholded_true, thresholded_pred)
        return loss



class ThresholdedVGGFeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = keras.Model(vgg.input, layer_outputs, name="VGG")
        self.mae = keras.losses.MeanAbsoluteError()

    def make_threshold(self,labels,image):
        labels=labels[:,:,:,0] +labels[:,:,:,1] +labels[:,:,:,2] 
        boolenmask=tf.math.greater_equal(labels,  tf.constant(-2, dtype=tf.float32))
        shape = tf.shape(boolenmask)
        boolenmask=tf.reshape(boolenmask,(shape[0],shape[1],shape[2],1))
        boolenmask=tf.concat((boolenmask,boolenmask,boolenmask),axis=3)
        thresholded_img=tf.where(boolenmask,image, tf.constant(0, dtype=tf.float32))
        return thresholded_img

    def call(self, y_true, y_pred):
        y,labels=y_true[0],y_true[1]
        thresholded_true=self.make_threshold(y,labels)
        thresholded_pred=self.make_threshold(y_pred,labels)
        thresholded_true = keras.applications.vgg19.preprocess_input(127.5 * (thresholded_true + 1))
        thresholded_pred = keras.applications.vgg19.preprocess_input(127.5 * (thresholded_pred + 1))
        real_features = self.vgg_model(thresholded_true)
        fake_features = self.vgg_model(thresholded_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss

class AdaptiveThresholdedFeatureLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = keras.losses.MeanAbsoluteError()

    def make_threshold(self,labels,image):
        boolenmask=tf.math.greater_equal(labels,  tf.constant(0, dtype=tf.float32))
        shape = tf.shape(boolenmask)
        boolenmask=tf.reshape(boolenmask,(shape[0],shape[1],shape[2],1))
        boolenmask=tf.concat((boolenmask,boolenmask,boolenmask),axis=3)
        thresholded_img=tf.where(boolenmask,image, tf.constant(-1, dtype=tf.float32))
        return thresholded_img


    def call(self, y_true, y_pred):
        y,labels=y_true[0],y_true[1]
        shape_label=tf.shape(labels)
        loss=float(0)
        for i in range(int(shape_label[3])):
            label=labels[:,:,:,i]
            weight=(tf.reduce_sum(label+tf.constant(1, dtype=tf.float32)))/2
            if weight!=0:
              if shape_label[0]!=None:
                weight=((float(shape_label[0])*float(shape_label[1])*float(shape_label[2]))/weight)
              else:
                weight=((float(shape_label[1])*float(shape_label[2]))/weight)

        thresholded_true=self.make_threshold(label,y)
        thresholded_pred=self.make_threshold(label,y_pred)
        loss_sub=self.mae(thresholded_true, thresholded_pred)
        loss_sub=loss_sub*weight
        loss=loss+loss_sub
        return loss

def get_numclasses(class_occurence):
  return np.float32((class_occurence> 0).sum())

def get_weightmap(coefficients,integers):
  return np.float32(coefficients[integers])

class OASISDiscriminatorLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy( from_logits=True)
        self.labelmix_function = keras.losses.MeanSquaredError()

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


    def get_class_balancing(self, label):
        class_occurence = tf.math.reduce_sum(label, axis=(0, 1, 2))
        class_occurence = tf.cast(class_occurence, dtype=tf.float32)
        num_of_classes=tf.numpy_function(get_numclasses, [class_occurence], tf.float32)
        coefficients = tf.cast(tf.math.reciprocal(class_occurence),dtype=tf.float32) * tf.cast(tf.size(label),dtype=tf.float32) / tf.cast((num_of_classes * label.shape[-1]),dtype=tf.float32)
        coefficients =tf.where(tf.math.is_inf(coefficients),tf.ones_like(coefficients),coefficients)
        integers = tf.math.argmax(label, axis=-1)
        integers=tf.expand_dims(integers, -1)
        weight_map = tf.numpy_function(get_weightmap, [coefficients,integers], tf.float32)
        return weight_map

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return tf.ones_like(input)
        else:
            return tf.zeros_like(input)

    def get_n1_target(self, input, label, target_is_real):
        targets = self.get_target_tensor(input, target_is_real)
        num_of_classes = label.shape[-1]
        integers = tf.math.argmax(label, axis=-1)
        targets = targets[:,:, :,0] * num_of_classes
        integers += tf.cast(targets, tf.int64 )
        integers = tf.clip_by_value(integers, clip_value_min=num_of_classes-1,clip_value_max=tf.cast(2e10, tf.int64 )) - num_of_classes + 1
        return tf.cast(integers,dtype=tf.float32)

    def call(self, inputs,label ):
        input,for_real=inputs[0],inputs[1]
        weight_map = self.get_class_balancing(label)
        target = self.get_n1_target(input, label, for_real)
        #--- n+1 loss ---
        loss = self.cross_entropy(target,input)

        if for_real:
            loss = loss *weight_map[:,:,:,0]
        
        return tf.reduce_mean(loss)
    
class DiscriminatorLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hinge_loss = keras.losses.Hinge()

    def call(self, y, is_real):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(label, y)