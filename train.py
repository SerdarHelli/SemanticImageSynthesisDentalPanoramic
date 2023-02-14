# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:34:33 2022

@author: pc
"""

import argparse    
from  networks.models import * 
from data.datagenerator import *
from metrics import *
from callbacks import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
import os
import numpy as np
import random
import sys 

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--train_data_path", type=str, required=True,help="Original Train Data Path of Tufs Dental Database")
parser.add_argument("--val_data_path", type=str, required=True,help="Original Val Data Path of Tufs Dental Database")
parser.add_argument("--seed", type=int, required=False,default=1234,help="Seed")
parser.add_argument("--model", type=str, required=True,default="oasis",help="Model OASIS or Gaugan. Two valid parametres -->'oasis' or 'gaugan'")
parser.add_argument("--epochs", type=int, required=False,default=35,help="Epochs")
parser.add_argument("--batch_size", type=int, required=False,default=1,help="Batch Size")
parser.add_argument("--disc_lr", type=int, required=False,default=1e-4,help="Discriminator Learning Rate")
parser.add_argument("--gen_lr", type=int, required=False,default=4e-4,help="Generator Learning Rate")
parser.add_argument("--shuffle", type=bool, required=False,default=True,help="Data Shuffle")
parser.add_argument("--data_flip", type=bool, required=False,default=True,help="Flip Data Augmentation")
parser.add_argument("--img_size", type=int, required=False,default=256,help="Image Size")
parser.add_argument("--latent_dim", type=int, required=False,default=32,help="Latent Dimension Size --> For Gaugan It should be 256. For OASIS It should be 32. ")
parser.add_argument("--vgg_feature_loss_coeff", type=int, required=False,default=0.1,help="VGG Feature Loss Coefficient")
parser.add_argument("--lambda_labelmix", type=int, required=False,default=10,help="LabelMix Loss Coefficient")
parser.add_argument("--feature_loss_coeff", type=int, required=False,default=10,help="Feature Loss Coefficient")
parser.add_argument("--kl_divergence_loss_coeff", type=int, required=False,default=0.1,help="KL Divergence  Loss Coefficient")
parser.add_argument("--save_perepoch", type=int, required=False,default=5,help="Save Checkpoint per epoch")
parser.add_argument("--logs_path", type=str, required=False,default="./training",help="Path Logs and  CheckPoints ")
parser.add_argument("--max_to_keep", type=int, required=False,default=None,help="How Many CheckPoint Max To Keep ")
parser.add_argument("--include_abnormality", type=bool, required=False,default=True,help="Include Abnormality ")
parser.add_argument("--disc_init_filters", type=int, required=False,default=16,help="Discriminator Init Filters")
parser.add_argument("--special_checkpoint", type=str, required=False,default=None,help="Spesific Checkpoint Path - Example './training/checpoint/ckpt-10' ")

args = vars(parser.parse_args())



random.seed(args["seed"])
np.random.seed(args["seed"])
tf.random.set_seed(args["seed"])

if not os.path.isdir(args["logs_path"]):
    os.makedirs(args["logs_path"])

if args["img_size"]!=256:
    print("\n")
    print("WARNING:Please Check the generator model.Beacuse It can have more or less upsample layer for {}".format(args["img_size"]))
    print("\n")
    sys.exit()



train_generator=DataGenerator(file_path=args["train_data_path"],batch_size=args["batch_size"],
                              img_dim=args["img_size"],data_flip=args["data_flip"],shuffle=args["shuffle"],with_abnormality=args["include_abnormality"])

 
val_generator=DataGenerator(file_path=args["val_data_path"],batch_size=args["batch_size"],
                              img_dim=args["img_size"],data_flip=False,shuffle=args["shuffle"],with_abnormality=args["include_abnormality"])

print("Caching Data...\n")
print("Train Data...\n")
train_dataset=train_generator.get_dataset()
print("Validation Data...\n")
val_dataset=val_generator.get_dataset()

if args["include_abnormality"]:
    num_classes=5
else:
    num_classes=4



if args["model"]=="oasis":
    print("OASISGAN is initialized\n")
    
    gaugan = OASISGauGAN(image_size=args["img_size"], num_classes=num_classes, batch_size=args["batch_size"], latent_dim=args["latent_dim"],special_checkpoint=args["special_checkpoint"],
                         vgg_feature_loss_coeff=args["vgg_feature_loss_coeff"],lambda_labelmix=args["lambda_labelmix"],
                         gen_lr=args["gen_lr"],disc_lr=args["disc_lr"],checkpoint_path=args["logs_path"],max_to_keep=args["max_to_keep"],disc_init_filters=args["disc_init_filters"])
    noise_mode="3D"

    
elif args["model"]=="gaugan":
    print("GauGAN is initialized\n")

    gaugan = GauGAN(image_size=args["img_size"], num_classes=num_classes, batch_size=args["batch_size"], latent_dim=args["latent_dim"],special_checkpoint=args["special_checkpoint"],
                         vgg_feature_loss_coeff=args["vgg_feature_loss_coeff"],kl_divergence_loss_coeff=args["kl_divergence_loss_coeff"],
                         gen_lr=args["gen_lr"],disc_lr=args["disc_lr"],checkpoint_path=args["logs_path"],max_to_keep=args["max_to_keep"],disc_init_filters=args["disc_init_filters"])
    noise_mode="1D"
    
else:
    print("Invalid Model Name...")
    sys.exit()



log_dir = args["logs_path"]+"/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


sample_label=next(iter(val_dataset))[3]

if noise_mode=="3D":
    sample_noise=tf.random.normal(shape=(gaugan.batch_size, gaugan.image_size*gaugan.image_size*gaugan.latent_dim),dtype=tf.float32)
    sample_noise=tf.reshape(sample_noise,(gaugan.batch_size, gaugan.image_size,gaugan.image_size,gaugan.latent_dim))
elif noise_mode=="1D":
    sample_noise=tf.random.normal(shape=( gaugan.batch_size,gaugan.latent_dim),dtype=tf.float32)



gaugan.compile()

history = gaugan.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args["epochs"],
    callbacks=[SaveCheckpoint(number_epoch=args["epochs"],per_epoch=args["save_perepoch"]),tensorboard_callback,SaveOneSample(sample_label,sample_noise,args["logs_path"])]
)
    
    
if train_dataset:
    del train_dataset


csv_path=os.path.join(args["logs_path"],"logs.csv")
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
else:
  df=pd.DataFrame(columns=[ 'disc_loss','gen_loss','vgg_loss','gen_disc_loss','Structure Similarity','val_disc_loss','val_gen_loss','val_vgg_loss','val_gen_disc_loss','val_Structure Similarity'])

if 'Unnamed: 0' in df:
  df.drop('Unnamed: 0', axis=1, inplace=True)


for i in range(len(history.history["disc_loss"])):
    data={'disc_loss':history.history["disc_loss"][i],'gen_loss':history.history["gen_loss"][i],'vgg_loss':history.history["vgg_loss"][i],
          'gen_disc_loss':history.history["gen_disc_loss"][i],'Structure Similarity':history.history["Structure Similarity"][i],
          'val_disc_loss':history.history["val_disc_loss"][i],'val_gen_loss':history.history["val_gen_loss"][i],'val_vgg_loss':history.history["val_vgg_loss"][i],
          'val_gen_disc_loss':history.history["val_gen_disc_loss"][i],'val_Structure Similarity':history.history["val_Structure Similarity"][i]}
    df=df.append(data,ignore_index=True)

df.to_csv(csv_path)

