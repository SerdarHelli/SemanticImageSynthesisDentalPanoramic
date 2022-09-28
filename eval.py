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

parser = argparse.ArgumentParser(prog="Eval")
parser.add_argument("--test_data_path", type=str, required=True,help="Original Test Data Path of Tufs Dental Database")
parser.add_argument("--seed", type=int, required=False,default=1234,help="Seed")
parser.add_argument("--model", type=str, required=True,default="oasis",help="Model OASIS or Gaugan. Two valid parametres -->'oasis' or 'gaugan'")
parser.add_argument("--batch_size", type=int, required=False,default=1,help="Batch Size")
parser.add_argument("--img_size", type=int, required=False,default=256,help="Image Size")
parser.add_argument("--latent_dim", type=int, required=False,default=32,help="Latent Dimension Size --> For Gaugan It should be 256. For OASIS It should be 32. ")
parser.add_argument("--logs_path", type=str, required=False,default="./training",help="Path Logs and  CheckPoints ")
parser.add_argument("--include_abnormality", type=bool, required=False,default=True,help="Include Abnormality ")
parser.add_argument("--special_checkpoint", type=str, required=False,default=None,help="Spesific Checkpoint Path - Example './training/checpoint/ckpt-10' ")
parser.add_argument("--inception_mode", type=str, required=False,default="v4",help="inception_mode - 'v4' common one - 'v3' keras based ")


args = vars(parser.parse_args())



random.seed(args["seed"])
np.random.seed(args["seed"])
tf.random.set_seed(args["seed"])


if not os.path.isdir(args["logs_path"]):
    print("Invalid Log Path. Check out it or Train it !")
    sys.exit()


if args["img_size"]!=256:
    print("\n")
    print("WARNING:Please Check the generator model.Beacuse It can have more or less upsample layer for {}".format(args["img_size"]))
    print("\n")
    sys.exit()




if args["include_abnormality"]:
    num_classes=5
else:
    num_classes=4


disc_lr=1e-4
gen_lr=4e-4
vgg_feature_loss_coeff=0.1
lambda_labelmix=10
feature_loss_coeff=10
kl_divergence_loss_coeff=0.1
save_perepoch=5
max_to_keep=None

if args["model"]=="oasis":
    print("OASISGAN is initialized\n")
    
    gaugan = OASISGauGAN(image_size=args["img_size"], num_classes=num_classes, batch_size=args["batch_size"], latent_dim=args["latent_dim"],
                         vgg_feature_loss_coeff=vgg_feature_loss_coeff,lambda_labelmix=lambda_labelmix,special_checkpoint=args["special_checkpoint"],
                         gen_lr=gen_lr,disc_lr=disc_lr,checkpoint_path=args["logs_path"],max_to_keep=max_to_keep,disc_init_filters=16)
    noise_mode="3D"

    
elif args["model"]=="gaugan":
    print("GauGAN is initialized\n")

    gaugan = GauGAN(image_size=args["img_size"], num_classes=num_classes, batch_size=args["batch_size"], latent_dim=args["latent_dim"],
                         vgg_feature_loss_coeff=vgg_feature_loss_coeff,kl_divergence_loss_coeff=kl_divergence_loss_coeff,special_checkpoint=args["special_checkpoint"],
                         gen_lr=gen_lr,disc_lr=disc_lr,checkpoint_path=args["logs_path"],max_to_keep=max_to_keep,disc_init_filters=16)
    noise_mode="1D"
    
else:
    print("Invalid Model Name...")
    sys.exit()

gaugan.compile(usage="eval")


test_generator=DataGenerator(file_path=args["test_data_path"],batch_size=args["batch_size"],
                              img_dim=args["img_size"],data_flip=False,shuffle=False,with_abnormality=args["include_abnormality"])

save_path_test=os.path.join(args["logs_path"],"test_results")
if not os.path.isdir(save_path_test):
    os.makedirs(save_path_test)

print("Test Evolution....\n")
metrics_test=GauaganMetrics(gaugan,test_generator,save_path=save_path_test,noise_mode=noise_mode,labels_mode="one_hot",inception_mode=args["inception_mode"])
print("Test Eval :\n ")
print("------------------------------------\n")
metrics_test.get_metrics_score(mask_apply=True)
print("No Applied Mask\n")
metrics_test.get_metrics_score(mask_apply=False)