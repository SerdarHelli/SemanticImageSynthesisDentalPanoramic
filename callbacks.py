
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime



class SaveOneSample(tf.keras.callbacks.Callback):
      def __init__(self, label,noise,path):
        super(SaveOneSample, self).__init__()
        self.label = label
        self.noise=noise
        self.path=os.path.join(path,"imgs")
        if not os.path.isdir(self.path):
          os.makedirs(self.path)

      def on_epoch_end(self, epoch, logs=None):

        # Generate the fake images.
        fake_images = self.model.generator([self.noise, self.label])
        sample=np.uint8(((fake_images[-1]*127.5)+127.5))
        sample_path=os.path.join(self.path,"{}_img_{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),epoch))
        plt.imsave(sample_path,sample)


class SaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,number_epoch, per_epoch=5):
        super(SaveCheckpoint, self).__init__()
        self.per_epoch = per_epoch
        self.number_epoch=number_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch%self.per_epoch)==0 and epoch!=0:
            self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)
        if self.number_epoch==epoch:
            self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)


