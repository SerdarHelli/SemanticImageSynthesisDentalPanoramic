# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:31:00 2022

@author: pc
"""

import shutil
import tensorflow as tf 
import numpy as np
import cv2
from natsort import natsorted
import os
import sys
import random

def convert_onechannel(img):
  if len(img.shape)>2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img= cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
  return img

def check_channel(img):
  if img.shape[2]>3:
    img=img[:,:,:4]
  return img

def convert_tobgr(img):
  if len(img.shape)<2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img



class DataGenerator():
    def __init__(self, file_path,batch_size,img_dim=256,data_flip=True,shuffle=True,with_abnormality=True,return_filesname=False):
        self.batch_size=batch_size
        self.data_path = file_path
        self.segm_zippath=os.path.join(file_path, "Segmentation.zip")
        self.expert_zippath=os.path.join(file_path, "Expert.zip")
        self.radio_zippath=os.path.join(file_path, "Radiographs.zip")
        self.img_dim=img_dim
        self.shuffle=shuffle
        self.data_flip=data_flip
        self.with_abnormality=with_abnormality
        self.path_img=os.path.join(file_path, "Radiographs")
        self.path_label_teeth=os.path.join(file_path, "Segmentation/teeth_mask")
        self.path_label_mandibular=os.path.join(file_path, "Segmentation/maxillomandibular")
        self.path_label_abnormal=os.path.join(file_path, "Expert/mask")
        self.return_filesname=return_filesname

        missing_data_error="Check your data path . It needs zips files or their folder of the data."  
        if not os.path.isdir(file_path):
            print("ERROR : Invalid Data Path : {} .. Exiting...".format(file_path))
            sys.exit()
        if not os.path.isfile(self.segm_zippath) :
            error="ERROR :Segmentation  zip file  {} is missing.. Exiting...".format(self.segm_zippath)
            if not os.path.isdir(self.path_label_teeth) or not os.path.isdir(self.path_label_mandibular):
              print(error)
              print("ERROR : Segmentation  folder   {} or {} is missing.. Exiting...".format(self.path_label_teeth,self.path_label_mandibular))
              print(missing_data_error)
              sys.exit()
        if not os.path.isfile(self.expert_zippath)  :
            error="ERROR :Expert  zip file   {} is missing.. Exiting...".format(self.expert_zippath)
            if not os.path.isdir(self.path_label_abnormal):
              print(error)
              print("ERROR :Expert  folder  {} is missing.. Exiting...".format(self.path_label_abnormal))
              print(missing_data_error)
              sys.exit()
        if not os.path.isfile(self.radio_zippath):
            error="ERROR :Radiographs zip file {} is missing. Exiting...".format(self.radio_zippath)
            if not os.path.isdir(self.path_img):
                print(error)
                print("ERROR :Radiographs folder {}  is missing. Exiting...".format(self.path_img))
                print(missing_data_error)
                sys.exit()   
        
   
        
        if not os.path.isdir(self.path_label_teeth) or not os.path.isdir(self.path_label_mandibular):
          shutil.unpack_archive(self.segm_zippath,file_path)

        if not os.path.isdir(self.path_label_abnormal):
          shutil.unpack_archive(self.expert_zippath,file_path)
        
        if not os.path.isdir(self.path_img):
          shutil.unpack_archive(self.radio_zippath,file_path)
        
        self.dirs_label_teeth=natsorted(os.listdir(self.path_label_teeth))

    def read_label(self,path,size):
      
      label = cv2.imread(path)
      label=convert_onechannel(label)
      label=cv2.resize(label, (size, size), interpolation= cv2.INTER_LINEAR )
      label=np.reshape(label,(size,size,1)) 
      return label
  
    def read_img(self,path,size):

      img = cv2.imread(path)
      img=convert_tobgr(img)
      img=check_channel(img)
      img=cv2.resize(img, (size, size), interpolation= cv2.INTER_LINEAR )
      img=np.reshape(img,(size,size,3))
      img=np.float32((img - 127.5) / 127.5 )
      return img
    
    def len(self):
        return len(self.dirs_label_teeth)
    
    def make_categoricalonehotlabelmap(self,mandibular,teeth,abnormal):

        categoricallabelmap=np.ones((self.img_dim,self.img_dim,1))

        mandibular=(mandibular/255)<1
        categoricallabelmap=np.where(mandibular,categoricallabelmap,2)

        teeth=(teeth/255)<1
        categoricallabelmap=np.where(teeth,categoricallabelmap,3)

        if self.with_abnormality:
          abnormal=(abnormal/255)<1
          categoricallabelmap=np.where(abnormal,categoricallabelmap,4)
          categoricallabelmap=tf.one_hot(categoricallabelmap, 5)
          categoricallabelmap=np.reshape(categoricallabelmap,(self.img_dim,self.img_dim,5)) 
        else:
          categoricallabelmap=tf.one_hot(categoricallabelmap, 4)
          categoricallabelmap=np.reshape(categoricallabelmap,(self.img_dim,self.img_dim,4)) 
        return np.float32(categoricallabelmap)

    def apply_flip(self,image,segmentationmap,label,categoricallabelmap):

        flipped_image=np.fliplr(image)
        flipped_segmentationmap=np.fliplr(segmentationmap)
        flipped_label=np.fliplr(label)
        flipped_categoricallabelmap=np.fliplr(categoricallabelmap)
        return flipped_image,flipped_segmentationmap,flipped_label,flipped_categoricallabelmap

  
    def make_segmentationmap(self,categorical_map):
        segmentationmap=np.zeros((self.img_dim,self.img_dim,3))
        segmentationmap[:,:,0]=categorical_map[:,:,3]
        if self.with_abnormality:
          segmentationmap[:,:,1]=categorical_map[:,:,4]          
        segmentationmap[:,:,2]=categorical_map[:,:,2]
        segmentationmap=np.float32((segmentationmap - 0.5) /0.5 )
        return segmentationmap

    def get_dataset(self):
        files_names=self.dirs_label_teeth
        images=[]
        segmentationmaps=[]
        categoricallabelmaps=[]
        labels=[]
        img_names=[]
        for i in range(len(files_names)):

            path_img=os.path.join(self.path_img,files_names[i].upper())
            path_label_teeth=os.path.join(self.path_label_teeth,files_names[i])
            path_label_mandibular=os.path.join(self.path_label_mandibular,files_names[i])
            path_abnormal=os.path.join(self.path_label_abnormal,files_names[i].upper())
            image=self.read_img(path_img,self.img_dim)
            images.append(image)
            teeth=self.read_label(path_label_teeth, self.img_dim)
            mandibular=self.read_label(path_label_mandibular, self.img_dim)
            abnormal=self.read_label(path_abnormal, self.img_dim)
            categorical_map=self.make_categoricalonehotlabelmap(mandibular, teeth, abnormal)
            categoricallabelmaps.append(categorical_map)
            segmentation_map=self.make_segmentationmap(categorical_map)
            segmentationmaps.append(segmentation_map)
            label=np.concatenate((teeth, mandibular),axis=2)
            img_names.append(files_names[i])

            if self.with_abnormality:
              label=np.concatenate((label,abnormal),axis=2)

            label=np.float32((label - 127.5) / 127.5 )
            labels.append(label)

            if self.data_flip:
              flipped_image,flipped_segmentationmap,flipped_label,flipped_categoricallabelmap=self.apply_flip(image,segmentation_map,label,categorical_map)
              images.append(flipped_image)
              categoricallabelmaps.append(flipped_categoricallabelmap)
              segmentationmaps.append(flipped_segmentationmap)
              labels.append(flipped_label)
              img_names.append(files_names[i])


        if self.return_filesname:
            dataset=tf.data.Dataset.from_tensor_slices((images,segmentationmaps,labels,categoricallabelmaps,img_names)).batch(self.batch_size)
        else: 
            dataset=tf.data.Dataset.from_tensor_slices((images,segmentationmaps,labels,categoricallabelmaps)).batch(self.batch_size)
            

        if self.shuffle:
          dataset=dataset.shuffle(200)

        return dataset
        