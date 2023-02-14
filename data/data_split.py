# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:31:41 2022

@author: pc
"""

import os
import shutil
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(prog="Data Split")
parser.add_argument("--data_path", type=str, required=False,help=" Orignal Data Path")
parser.add_argument("--train_data_path", type=str, required=False,default="./Tufs_Raw_Train",help=" Train Data Path  of Extraction")
parser.add_argument("--val_data_path", type=str, required=False,default="./Tufs_Raw_Val",help="Val Data Path of Extraction ")
parser.add_argument("--test_data_path", type=str, required=False,default="./Tufs_Raw_Test",help=" Val Data Path of Extraction")

args = vars(parser.parse_args())


path=args["data_path"]
path_val=args["val_data_path"]
path_train=args["train_data_path"]
path_test=args["test_data_path"]

df=pd.read_csv("./data_split.csv")
    
if 'Unnamed: 0' in df:
  df.drop('Unnamed: 0', axis=1, inplace=True) 

   
segm_zippath=os.path.join(path, "Segmentation.zip")
expert_zippath=os.path.join(path, "Expert.zip")
radio_zippath=os.path.join(path, "Radiographs.zip")

if not os.path.isdir(path):
    print("Data folder path {} does not exist. Exiting...".format(path))
    sys.exit()
if not os.path.isfile(segm_zippath):
    print("Segmentation zip file  {} does not exist. Exiting...".format(segm_zippath))
    sys.exit()
if not os.path.isfile(expert_zippath):
    print("Expert zip file {} does not exist. Exiting...".format(expert_zippath))
    sys.exit()
if not os.path.isfile(radio_zippath):
    print("Radiographs zip file {} does not exist. Exiting...".format(radio_zippath))
    sys.exit()   

path_teeth=os.path.join(path,"Segmentation/teeth_mask/")
path_Radiographs=os.path.join(path,"Radiographs/")
path_maxillomandibular=os.path.join(path,"Segmentation/maxillomandibular/")
path_expert=os.path.join(path,"Expert/mask/")

if not os.path.isdir(path_teeth) or not os.path.isdir(path_maxillomandibular):
  shutil.unpack_archive(segm_zippath,path)

if not os.path.isdir(path_expert):
  shutil.unpack_archive(expert_zippath,path)

if not os.path.isdir(path_Radiographs):
  shutil.unpack_archive(radio_zippath,path)




path_train_teeth=os.path.join(path_train,"Segmentation/teeth_mask/")
path_test_teeth=os.path.join(path_test,"Segmentation/teeth_mask/")
path_val_teeth=os.path.join(path_val,"Segmentation/teeth_mask/")

path_train_Radiographs=os.path.join(path_train,"Radiographs/")
path_test_Radiographs=os.path.join(path_test,"Radiographs/")
path_val_Radiographs=os.path.join(path_val,"Radiographs/")

path_train_maxillomandibular=os.path.join(path_train,"Segmentation/maxillomandibular/")
path_test_maxillomandibular=os.path.join(path_test,"Segmentation/maxillomandibular/")
path_val_maxillomandibular=os.path.join(path_val,"Segmentation/maxillomandibular/")


path_train_expert=os.path.join(path_train,"Expert/mask/")
path_test_expert=os.path.join(path_test,"Expert/mask/")
path_val_expert=os.path.join(path_val,"Expert/mask/")


def split_onedir(df,path_train,path_test,path_val,path):
    make_dirs(path_train,path_test,path_val)
    for i in range(len(df)):
        if df["usage"][i]=="train":
            img_name=df["img_name"][i].lower()
            shutil.copy(path+img_name,path_train+img_name)
        if df["usage"][i]=="test":
            img_name=df["img_name"][i]
            shutil.copy(path+img_name,path_test+img_name)
        if df["usage"][i]=="val":
            img_name=df["img_name"][i]
            shutil.copy(path+img_name,path_val+img_name)      
    dirs_train=os.listdir(path_train)
    dirs_test=os.listdir(path_test)
    dirs_val=os.listdir(path_val)
    
    for x in (dirs_train):
        if x in dirs_test:
            print("Error Check your data ,There is duplicated")
    for y in (dirs_train):
        if y in dirs_val:
            print("Error Check your data ,There is duplicated")
            
            
            
            
def make_dirs(path_train,path_test,path_val):
    if not os.path.isdir(path_train):
         os.makedirs(path_train)
    if not os.path.isdir(path_test):
         os.makedirs(path_test)
    if not os.path.isdir(path_val):
         os.makedirs(path_val)
         
         
split_onedir(df,path_train_teeth,path_test_teeth,path_val_teeth,path_teeth)             
split_onedir(df,path_train_Radiographs,path_test_Radiographs,path_val_Radiographs,path_Radiographs)
split_onedir(df,path_train_maxillomandibular,path_test_maxillomandibular,path_val_maxillomandibular,path_maxillomandibular)
split_onedir(df,path_train_expert,path_test_expert,path_val_expert,path_expert)      
   
