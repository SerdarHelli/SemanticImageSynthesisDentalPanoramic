# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:25:45 2022

@author: pc
"""

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import sys
import cv2
import os
from inception import *
import pandas as pd



class GauaganMetrics():
      def __init__(self, gaugan_model,data_generator,save_path=None,noise_mode="3D",labels_mode="one_hot",inception_mode="v4"):

          #gaugan_model : Gaugan Model Class with loadded generator
          #data_generator : Defined Data Generator Class - Check Data Generator Class
          #noise_mode - "3D" or "1D" Defining Dimesion of Noise as input to generator
          #labels_mode - "one_hot" or "binary" Defining which label type  as input to generator
          #inception_mode - "v4" common one - "v3" keras based
          self.gaugan_model=gaugan_model
         
          self.data_generator=data_generator
          self.labels_mode=labels_mode
          self.noise_mode=noise_mode
          self.dataset=data_generator.get_dataset()
          self.inception_mode=inception_mode
          if self.inception_mode=="v3":
              self.inception_model=InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
          self.images=[]
          self.segmentation_maps=[]
          self.labels=[]
          self.onehotlabels=[]
          self.org_shapes=[]
          self.load_data()
          self.save_path=save_path
          if save_path!=None:
            self.real_save_path=os.path.join(self.save_path,"reals")
            if not os.path.isdir(self.real_save_path):
              os.makedirs(self.real_save_path)
            self.fake_save_path=os.path.join(self.save_path,"fakes")
            if not os.path.isdir(self.fake_save_path):
              os.makedirs(self.fake_save_path)


      def load_data(self):
          iterator = iter(self.dataset)
          for i in range(self.data_generator.len()):
            data = next(iterator)
            for j in range(self.data_generator.batch_size):
              self.images.append(data[0][j])
              self.segmentation_maps.append(data[1][j])
              self.labels.append(data[2][j])
              self.onehotlabels.append(data[3][j])

      def predict(self):
          predict_list=[]
          if self.labels_mode=="one_hot":
            data_list=np.asarray(self.onehotlabels)
          elif self.labels_mode=="binary":
            data_list=np.asarray(self.labels)
          else:
            print("Invalid Labels Mode")
            sys.exit()
          for i in  range (len(data_list)):
              x=data_list[i,:,:,:]
              x=np.reshape(x,(1,x.shape[0],x.shape[1],x.shape[2]))
              if self.noise_mode=="3D":
                latent_vector=tf.random.normal(shape=(1,self.gaugan_model.image_size *self.gaugan_model.image_size*self.gaugan_model.latent_dim),dtype=tf.float32)
                latent_vector=tf.reshape(latent_vector,(1, self.gaugan_model.image_size,self.gaugan_model.image_size,self.gaugan_model.latent_dim))
                model_name="oasis"
              elif self.noise_mode=="1D":
                latent_vector=tf.random.normal(shape=(1,self.gaugan_model.latent_dim),dtype=tf.float32)
                model_name="gaugan"
              else:
                print("Invalid Noise Mode")
                sys.exit()  
              predict=self.gaugan_model.generator.predict([latent_vector,x])
              predict_list.append(predict[-1])
              if self.save_path!=None:
                fake_img=np.uint8((predict[-1]*127.5)+127.5)
                fake_img=cv2.resize(fake_img, (512, 256), interpolation= cv2.INTER_LANCZOS4)
                fake_img = cv2.fastNlMeansDenoisingColored(fake_img,None,3,3,7,5)
                real_img=np.uint8((self.images[i]*127.5)+127.5)
                real_img=cv2.resize(real_img, (512, 256), interpolation= cv2.INTER_LANCZOS4)

                cv2.imwrite(os.path.join(self.real_save_path,(str(i)+"_{}_real.png".format(model_name))),real_img)
                cv2.imwrite(os.path.join(self.fake_save_path,(str(i)+"_{}fake.png".format(model_name))),fake_img)

          return np.asarray(predict_list)
        

      def scale_images(self,images):
          images_list = list()
          new_shape=(299,299,3)
          for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
          return np.asarray(images_list)
      

      def get_activations_v3(self,images):
          images = self.scale_images(images)
          images = preprocess_input(images)
          activations = self.inception_model.predict(images)
          return activations



      # calculate frechet inception distance
      def calculate_fid(self, images1, images2,eps=1e-6):
          
          # calculate activations
          if self.inception_mode=="v3":
              act1 = self.get_activations_v3(images1)
              act2 = self.get_activations_v3(images2)
          elif self.inception_mode=="v4":
              act1=get_activations_fromarray_v4(images1,pth_inception=None,batch_size=50)
              act2=get_activations_fromarray_v4(images2,pth_inception=None,batch_size=50)
          else:
              print("Invalid Inception Mode")
              sys.exit()  

          # calculate mean and covariance statistics
          mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
          mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
          # calculate sum squared difference between means
          assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
          assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
          diff = mu1 - mu2

          #ssdiff = np.sum((mu1 - mu2)**2.0)
          # calculate sqrt of product between cov
          covmean,_ = sqrtm(sigma1.dot(sigma2), disp=False)
          if not np.isfinite(covmean).all():
              msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
              print(msg)
              offset = np.eye(sigma1.shape[0]) * eps
              covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

          # check and correct imaginary numbers from sqrt
          if np.iscomplexobj(covmean):
            covmean = covmean.real
          # calculate score
          tr_covmean=np.trace(covmean)
          fid = diff.dot(diff) + np.trace(sigma1)+ np.trace(sigma2) - 2.0 * tr_covmean
          return fid

      def make_threshold(self,list_imgs):
          res_list=[]
          array_segmenationmaps=np.asarray(self.segmentation_maps)
          for i in range(len(list_imgs)):
            a=np.uint8((array_segmenationmaps[i,:,:,:]*127.5)+127.5)
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY)
            img=np.uint8((list_imgs[i,:,:,:]*127.5)+127.5)
            res = cv2.bitwise_and(img,img,mask = mask)
            res_list.append(res)
          return np.asarray(res_list)
        

      def convert_uint8(self,list_imgs):
          res_list=[]
          for i in range(len(list_imgs)):
              img=np.uint8((list_imgs[i,:,:,:]*127.5)+127.5)
              res_list.append(img)
          return np.asarray(res_list)

      def save_ssim(self,loss_ssim,name):
          ssim=list(loss_ssim.numpy())
          df=pd.DataFrame(data={"ssim":ssim})
          path=os.path.join(self.save_path,"{}_results_ssim.csv".format(name))
          df.to_csv(path, index=False)


      def get_metrics_score(self,mask_apply=True):
          ground_truth=np.asarray(self.images)

          generated_images=self.predict()   
          if mask_apply:    
            x_generated_images=self.make_threshold(generated_images)
            x_ground_truth=self.make_threshold(ground_truth)
            print("Masks Applied")
            ssim_name="maskapplied"
          else:
            x_generated_images=self.convert_uint8(generated_images)
            x_ground_truth=self.convert_uint8(ground_truth)
            ssim_name="nomask"


          print('Prepared', x_ground_truth.shape, x_generated_images.shape)
          # convert integer to floating point values
          images1 = x_ground_truth.astype('float32')
          images2 = x_generated_images.astype('float32')
          # resize images

          # fid between images1 and images2
          self.fid = self.calculate_fid(images1, images2)
          self.loss_ssim = tf.image.ssim(x_ground_truth, x_generated_images ,max_val=255, filter_size=11,
                                    filter_sigma=1.5, k1=0.01, k2=0.03)
          self.save_ssim(self.loss_ssim,ssim_name)
          print("------------------------------------\n")
          print('FID (different): %.3f' % self.fid)
          print("------------------------------------\n")
          print("Structre Similiratiy : " ,tf.reduce_mean(self.loss_ssim).numpy())

