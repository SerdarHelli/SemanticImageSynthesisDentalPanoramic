# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:15:47 2022

@author: pc
"""
from networks.layers import *
from networks.generator import *
from networks.discriminator import *
from networks.encoder import *
from networks.losses import *
import os
import tensorflow as tf
import datetime

class OASISGauGAN(keras.Model):
    def __init__(
        self,
        image_size,
        num_classes,
        batch_size,
        latent_dim,
        checkpoint_path,
        special_checkpoint,
        vgg_feature_loss_coeff=0.1,
        lambda_labelmix=10,
        gen_lr=1e-4,
        disc_lr=4e-4,
        max_to_keep=None,
        disc_init_filters=16,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_shape = (image_size*2, image_size, 3)
        self.mask_shape = (image_size*2, image_size, num_classes)
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.gen_lr=gen_lr
        self.disc_lr=disc_lr
        self.discriminator = build_OASISdiscriminator(self.image_shape, init_filters=disc_init_filters,NUM_CLASSES=self.num_classes)
        self.generator = build_OASISgenerator(self.mask_shape,self.latent_dim)
        self.lambda_labelmix=lambda_labelmix
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
        self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
        self.gen_disc_loss_tracker = tf.keras.metrics.Mean(name="gen_disc_loss")
        self.ssim_tracker = tf.keras.metrics.Mean(name="Structure Similarity") 
        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
          os.makedirs(self.checkpoint_dir)
          print("New Checkpoint Folder Initialized...")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.max_to_keep=max_to_keep
        self.special_checkpoint=special_checkpoint

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.gen_loss_tracker,
            self.vgg_loss_tracker,
            self.gen_disc_loss_tracker,
            self.ssim_tracker

        ]


    def compile(self,usage="train", **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = keras.optimizers.Adam(
            self.gen_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            self.disc_lr, beta_1=0.0, beta_2=0.999
        )

        self. discriminator_loss = OASISDiscriminatorLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()
        self.ssim=StructureSmilarity()
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
    
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint,self.checkpoint_dir,max_to_keep=self.max_to_keep )
        #tf.train.Checkpoint.restore(...).expect_partial() ignore errors

        if self.ckpt_manager.latest_checkpoint and self.special_checkpoint==None:
            if usage=="eval":
                self.checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            else:
                self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            tf.print("-----------Restoring from {}-----------".format(
                self.ckpt_manager.latest_checkpoint))
        elif self.special_checkpoint!=None:
            if usage=="eval":
                self.checkpoint.restore(self.special_checkpoint).expect_partial()
            else:
                self.checkpoint.restore(self.special_checkpoint)
            tf.print("-----------Restoring from {}-----------".format(
                self.special_checkpoint))
        else:
            tf.print("-----------Initializing from scratch-----------")
  


    def train_discriminator(self, latent_vector, real_image, labels):
        fake_images = self.generator([latent_vector, labels])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator(fake_images)
            pred_real = self.discriminator(real_image)
            loss_fake = self.discriminator_loss([pred_fake, False],labels)
            loss_real = self.discriminator_loss([pred_real, True],labels)
            mixed_inp, mask = self.generate_labelmix(labels, fake_images, real_image)
            output_D_mixed = self.discriminator(mixed_inp)
            labelmix_regularization = self.lambda_labelmix * self.discriminator_loss.loss_labelmix(mask, output_D_mixed, pred_fake,
                                                                                pred_real)
            total_loss=loss_fake+loss_real+labelmix_regularization

        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(
            total_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss,fake_images

    def place_target(self,target_map):
        all_classes = np.unique(target_map)
        for c in all_classes:
          target_map[target_map == c] = np.random.randint(0,2,(1))
        return np.float32(target_map)


    def generate_labelmix(self,label, fake_image, real_image):  
        target_map = tf.math.argmax(label, axis = -1)
        target_map=tf.expand_dims(target_map, -1)
        target_map=tf.numpy_function(self.place_target, [target_map], tf.float32)
        mixed_image = target_map*real_image+(1-target_map)*fake_image
        return mixed_image, target_map


        
    def train_generator(
        self, latent_vector, labels, image
    ):

        #Generator learns through the signal provided by the discriminator. During
        #backpropagation, we only update the generator parameters.
        with tf.GradientTape() as tape:
            fake_image = self.generator(
                [latent_vector, labels]
            )
            fake_d_output=self.discriminator(fake_image)
            # Compute generator losses.
            d_loss = self.discriminator_loss([fake_d_output, True],labels)
            vgg_loss = (self.vgg_feature_loss_coeff )* self.vgg_loss(image, fake_image)

            total_loss = d_loss  + vgg_loss 
        all_trainable_variables = (
            self.generator.trainable_variables 
        )
        gradients = tape.gradient(total_loss, all_trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables)
        )
        return total_loss, d_loss,vgg_loss

    def train_step(self, data):
        image, _, _,labels = data
        latent_vector=tf.random.normal(shape=( self.batch_size,self.image_size*2*self.image_size*self.latent_dim),dtype=tf.float32)
        latent_vector=tf.reshape(latent_vector,(self.batch_size, self.image_size*2,self.image_size,self.latent_dim))

        discriminator_loss,fake_images = self.train_discriminator(
            latent_vector, image,labels)
        
        (generator_loss,d_loss,vgg_loss) = self.train_generator(
            latent_vector, labels, image)
        

        # Report progress.
        ssim=self.ssim(y_true=image,y_pred=fake_images)
        self.ssim_tracker.update_state(ssim)
        self.disc_loss_tracker.update_state(discriminator_loss)
        self.gen_loss_tracker.update_state(generator_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.gen_disc_loss_tracker.update_state(d_loss)
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, data):

        image, _, _,labels = data
        # Obtain the learned moments of the real image distribution.
        # Sample a latent from the distribution defined by the learned moments.
        latent_vector=tf.random.normal(shape=( self.batch_size,self.image_size*2*self.image_size*self.latent_dim),dtype=tf.float32)
        latent_vector=tf.reshape(latent_vector,(self.batch_size, self.image_size*2,self.image_size,self.latent_dim))

        # Generate the fake images.
        fake_images = self.generator([latent_vector, labels])
        # Calculate the losses.
        pred_fake = self.discriminator(fake_images)
        pred_real = self.discriminator(image)
        loss_fake = self.discriminator_loss([pred_fake, False],labels)
        loss_real = self.discriminator_loss([pred_real, True],labels)
        mixed_inp, mask = self.generate_labelmix(labels, fake_images, image)
        output_D_mixed = self.discriminator(mixed_inp)
        labelmix_regularization = self.lambda_labelmix * self.discriminator_loss.loss_labelmix(mask, output_D_mixed, pred_fake,pred_real)

        total_discriminator_loss=loss_fake+loss_real+labelmix_regularization

        fake_image = self.generator(
            [latent_vector, labels]
        )

        fake_d_output=self.discriminator(fake_image)
        vgg_loss = (self.vgg_feature_loss_coeff )* self.vgg_loss(image, fake_image)
        d_loss = self.discriminator_loss([fake_d_output, True],labels)

        total_generator_loss = d_loss  + vgg_loss 

        # Report progress
        ssim=self.ssim(y_true=image,y_pred=fake_images)
        self.ssim_tracker.update_state(ssim)
        self.disc_loss_tracker.update_state(total_discriminator_loss)
        self.gen_loss_tracker.update_state(total_generator_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.gen_disc_loss_tracker.update_state(d_loss)

        results = {m.name: m.result() for m in self.metrics}

        return results

    def call(self, inputs):

        latent_vectors, labels = inputs
        return self.generator([latent_vectors, labels])
    
    
    
    
    
    
    

class GauGAN(keras.Model):
    def __init__(
        self,
        image_size,
        num_classes,
        batch_size,
        latent_dim,
        checkpoint_path,
        special_checkpoint,
        feature_loss_coeff=10,
        vgg_feature_loss_coeff=0.1,
        kl_divergence_loss_coeff=0.1,
        gen_lr=1e-4, 
        disc_lr=4e-4,
        max_to_keep=None,
        disc_init_filters=16,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_shape = (image_size*2, image_size, 3)
        self.mask_shape = (image_size*2, image_size, num_classes)
        self.feature_loss_coeff = feature_loss_coeff
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.kl_divergence_loss_coeff = kl_divergence_loss_coeff
        self.gen_lr=gen_lr
        self.disc_lr=disc_lr
        self.discriminator = build_discriminator(self.image_shape,disc_init_filters)
        self.generator = build_generator(self.mask_shape,self.latent_dim)
        self.encoder = build_encoder(self.image_shape,encoder_downsample_factor=disc_init_filters, latent_dim=self.latent_dim) 
        self.sampler = GaussianSampler(batch_size, latent_dim)
        self.patch_size, self.combined_model = self.build_combined_generator()
        self.ssim_tracker = tf.keras.metrics.Mean(name="Structure Similarity") 
        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
          os.makedirs(self.checkpoint_dir)
          print("New CheckPoint Folder Initialized...")
        self.max_to_keep=max_to_keep
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_tracker = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.special_checkpoint=special_checkpoint

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.gen_loss_tracker,
            self.feat_loss_tracker,
            self.vgg_loss_tracker,
            self.kl_loss_tracker,
            self.ssim_tracker 
        ]

    def build_combined_generator(self):
        self.discriminator.trainable = False
        mask_input = keras.Input(shape=self.mask_shape, name="mask")
        image_input = keras.Input(shape=self.image_shape, name="image")
        latent_input = keras.Input(shape=(self.latent_dim), name="latent")
        generated_image = self.generator([latent_input, mask_input])
        discriminator_output = self.discriminator([image_input, generated_image])
        patch_size = discriminator_output[-1].shape[1]
        combined_model = keras.Model(
            [latent_input, mask_input, image_input],
            [discriminator_output, generated_image],
        )
        return patch_size, combined_model

    def compile(self,usage="train",  **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = keras.optimizers.Adam(
            self.gen_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            self.disc_lr, beta_1=0.0, beta_2=0.999
        )

        self. discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()
        self.ssim=StructureSmilarity()
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

#tf.train.Checkpoint.restore(...).expect_partial() ignore errors
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint,self.checkpoint_dir,max_to_keep=self.max_to_keep )
        if self.ckpt_manager.latest_checkpoint and self.special_checkpoint==None:
            if usage=="eval":
                self.checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            else:
                self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            tf.print("-----------Restoring from {}-----------".format(
                self.ckpt_manager.latest_checkpoint))
        elif self.special_checkpoint!=None:
            if usage=="eval":
                self.checkpoint.restore(self.special_checkpoint).expect_partial()
            else:
                self.checkpoint.restore(self.special_checkpoint)

            tf.print("-----------Restoring from {}-----------".format(
                self.special_checkpoint))
        else:
            tf.print("-----------Initializing from scratch-----------")
   


    def train_discriminator(self, latent_vector, segmentation_map, real_image, labels):
        fake_images = self.generator([latent_vector, labels])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator([segmentation_map, fake_images])[-1]
            pred_real = self.discriminator([segmentation_map, real_image])[-1]
            loss_fake = self.discriminator_loss(pred_fake, False)
            loss_real = self.discriminator_loss(pred_real, True)
            total_loss = 0.5 * (loss_fake + loss_real)
        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(
            total_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss,fake_images


    def train_generator(
        self, latent_vector, segmentation_map, labels, image, mean, variance
    ):
        # Generator learns through the signal provided by the discriminator. During
        # backpropagation, we only update the generator parameters.
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            real_d_output = self.discriminator([segmentation_map, image])
            fake_d_output, fake_image = self.combined_model(
                [latent_vector, labels, segmentation_map]
            )
            pred = fake_d_output[-1]
            # Compute generator losses.
            g_loss = generator_loss(pred)
            kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
            vgg_loss = (self.vgg_feature_loss_coeff )* self.vgg_loss(image, fake_image)
            feature_loss =(self.feature_loss_coeff ) * self.feature_matching_loss(
                real_d_output, fake_d_output
            )
            total_loss = g_loss + kl_loss + vgg_loss + feature_loss

        all_trainable_variables = (
            self.combined_model.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables)
        )
        return total_loss, feature_loss, vgg_loss, kl_loss
    

    def train_step(self, data):
        image, segmentation_map, _,labels = data
        mean, variance = self.encoder(image)
        latent_vector = self.sampler([mean, variance])

        discriminator_loss,fake_images = self.train_discriminator(
            latent_vector, segmentation_map, image, labels
        )
        (generator_loss, feature_loss, vgg_loss, kl_loss) = self.train_generator(
            latent_vector, segmentation_map, labels, image, mean, variance
        )
        # Report progress.
        ssim=self.ssim(y_true=image,y_pred=fake_images)
        self.ssim_tracker.update_state(ssim)
        self.disc_loss_tracker.update_state(discriminator_loss)
        self.gen_loss_tracker.update_state(generator_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, data):
        image, segmentation_map, _,labels = data
        # Obtain the learned moments of the real image distribution.
        mean, variance = self.encoder(image)


        # Sample a latent from the distribution defined by the learned moments.
        latent_vector = self.sampler([mean, variance])

        # Generate the fake images.
        fake_images = self.generator([latent_vector, labels])
        
        # Calculate the losses.
        pred_fake = self.discriminator([segmentation_map, fake_images])[-1]
        pred_real = self.discriminator([segmentation_map, image])[-1]
        loss_fake = self.discriminator_loss(pred_fake, False)
        loss_real = self.discriminator_loss(pred_real, True)
        total_discriminator_loss = 0.5 * (loss_fake + loss_real)
        real_d_output = self.discriminator([segmentation_map, image])
        fake_d_output, fake_image = self.combined_model(
            [latent_vector, labels, segmentation_map]
        )
        pred = fake_d_output[-1]
        g_loss = generator_loss(pred)
        kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
        vgg_loss = (self.vgg_feature_loss_coeff )* self.vgg_loss(image, fake_image)
        feature_loss =(self.feature_loss_coeff ) * self.feature_matching_loss(
                      real_d_output, fake_d_output
                  )


        total_generator_loss = g_loss + kl_loss + vgg_loss + feature_loss
        
        # Report progress.
        ssim=self.ssim(y_true=image,y_pred=fake_images)
        self.ssim_tracker.update_state(ssim)
        self.disc_loss_tracker.update_state(total_discriminator_loss)
        self.gen_loss_tracker.update_state(total_generator_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        latent_vectors, labels = inputs
        return self.generator([latent_vectors, labels])
    
    
    
    
    
    
    
