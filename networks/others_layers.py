import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_addons as tfa
from networks.layers import *

class WindowAttentionLayer(layers.Layer):
    def __init__(self,epsilon=1e-6,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=epsilon

    def build(self, input_shape): 
        self.filters=int(input_shape[3])
        self.windowpatches=WindowPatches(int(input_shape[1]))
        self.conv1=layers.Conv2D(self.filters//2,1,padding="same")
        self.conv2=layers.Conv2D(self.filters,3,padding="same")
        self.conv4=layers.Conv2D(self.filters,3,padding="same")
        self.spade1= SPADE(self.filters)

    def call(self,input,masks):
        windows=self.windowpatches(masks)
        windows=self.conv1(windows)
        windows=self.conv2(windows)
        out=tf.concat([input,windows],axis=3)
        out=self.conv4(out)
        out=self.spade1(out,masks)
        return tf.nn.leaky_relu(out,0.2)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class ResBlockMultiHeadAttention(layers.Layer):
    def __init__(self,epsilon=1e-6,mask_dim=256,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=epsilon
        self.mask_dim=mask_dim
        
    def build(self, input_shape): 
        self.window_patches=WindowPatches(int(input_shape[1]))
        self.filters=int(input_shape[3])
        self.patches=Patches(int(input_shape[1]))
        self.patch_encoder=PatchEncoder(((self.mask_dim // int(input_shape[1])) ** 2),self.filters//4)
        self.layer_norm1=layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm2=layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm3=layers.LayerNormalization(epsilon=self.epsilon)
        self.multi_head=layers.MultiHeadAttention(num_heads=2, key_dim=self.filters//4, dropout=0.1)
        self.add1=layers.Add()
        self.multi=layers.Multiply()
        self.globalav=layers.GlobalAveragePooling1D()
        self.conv1=layers.Conv2D(self.filters//4,1,padding="same")
        self.conv2=layers.Conv2D(self.filters,3,padding="same")
        self.spade1= SPADE(self.filters)

    def call(self,input,masks,segmentation_map):
        patches=self.patches(segmentation_map)
        encoded_patches=self.patch_encoder(patches)
        x1=self.layer_norm1(encoded_patches)
        attention_output=self.multi_head(x1,x1)
        x2=self.add1([attention_output,encoded_patches])
        representation=self.layer_norm2(x2)
        representation=self.globalav(representation)
        windows=self.window_patches(segmentation_map)
        windows=self.conv1(windows)
        attended=self.multi([windows,representation])
        conv=self.conv2(attended)
        conv=self.spade1(conv,masks)
        out=input+(tf.nn.leaky_relu(conv,0.2))
        return out



class TransformerLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads,rate=0.1):
        super(TransformerLayer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,dropout=rate)
        self.ffn = keras.Sequential(
            [layers.Dense(embed_dim*1, activation="relu"),layers.Dropout(rate), layers.Dense(embed_dim),layers.Dropout(rate)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x1= self.layernorm1(inputs)
        attention_output  = self.att(x1, x1)
        x2=inputs + attention_output
        x3 = self.layernorm1(x2)
        x3 = self.ffn(x3)
        output=x3+x2
        return output

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.embed_dim=embed_dim

    def build(self,input_shape):
        self.maxlen =int(input_shape[1])

        self.projection = layers.Dense(units = self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        #maxlen = tf.shape(x)[-1] # Normalde time sonda mı acaba???
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        output = self.pos_emb(positions)+self.projection(x)
        return output
        



class ResBlockFourierPositionalAttention(layers.Layer):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.gamma=10

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]
        self.filters=int(input_shape[3])
        self.f_positionalattention=FourierPositionalAttention(32,int(input_shape[1]),int(input_shape[1]))
        self.conv1= layers.Conv2D(self.filters, 4, padding="same",use_bias=False,kernel_initializer=tf.keras.initializers.GlorotNormal(),activation="sigmoid")
        self.conv3 = layers.Conv2D(self.filters, 3, padding="same")
        self.spade1 = SPADE(self.filters)
        self.dropout=layers.Dropout(0.25)

    def call(self,input,masks):
        encoded=(self.f_positionalattention(masks))*self.gamma
        alpha=self.conv1(encoded)
        x=tf.concat([alpha,input],axis=3)
        x=self.conv3(x)
        x=self.spade1(x,masks)
        x=self.dropout(tf.nn.leaky_relu(x,0.2))
        return x

class InSpadeMultiHeadAttentionLayer(layers.Layer):
    def __init__(self,epsilon=1e-6,mask_dim=256,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=epsilon
        self.mask_dim=mask_dim


    def build(self, input_shape): 
        self.filters=int(input_shape[3])
        self.patches=Patches(int(input_shape[1]))
        self.patch_encoder=PatchEncoder(((self.mask_dim // int(input_shape[1])) ** 2),self.filters)
        self.layer_norm1=layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm2=layers.LayerNormalization(epsilon=self.epsilon)
        self.layer_norm3=layers.LayerNormalization(epsilon=self.epsilon)
        self.multi_head=layers.MultiHeadAttention(num_heads=2, key_dim=self.filters, dropout=0.2)
        self.add1=layers.Add()
        self.multi=layers.Multiply()
        self.globalav=layers.GlobalAveragePooling1D()
        self.dropout1 = layers.Dropout(0.3)

    def call(self,input,segmentation):
        segmentation=(segmentation+1)/2
        patches=self.patches(segmentation)
        encoded_patches=self.patch_encoder(patches)
        x1=self.layer_norm1(encoded_patches)
        attention_output=self.multi_head(x1,x1)
        x2=self.add1([attention_output,encoded_patches])
        representation=self.layer_norm2(x2)
        representation=self.globalav(representation)
        attended=self.multi([input,representation])
        return attended



class UpSampleConv(layers.Layer):
    def __init__(self,filters,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters

    def build(self, input_shape): 
        self.convtranspose=layers.Conv2DTranspose(self.filters,(3,3),padding="same",strides=(2,2))
        self.spade= SPADE(self.filters)

    def call(self,input,masks):
        x=self.convtranspose(input)
        x=self.spade(x,masks)         
        return tf.nn.leaky_relu(x,0.2)
        
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



class WindowPatches(layers.Layer):
    def __init__(self, patch_size,epsilon=1e-6):
        super(WindowPatches, self).__init__()
        self.patch_size = patch_size
        self.epsilon=epsilon

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )


        num_patches=int((tf.shape(images)[1]*tf.shape(images)[1]) /(self.patch_size*self.patch_size))
        patches = tf.reshape(patches, [batch_size, num_patches, self.patch_size*self.patch_size,int(tf.shape(images)[3])])

        return patches


class PatchesToImage(layers.Layer):
    def __init__(self, imgh, imgw, imgc, patsz, is_squeeze=True, **kwargs):
        super(PatchesToImage, self).__init__(**kwargs)
        self.H = (imgh // patsz) * patsz
        self.W = (imgw // patsz) * patsz
        self.C = imgc
        self.P = patsz
        self.is_squeeze = is_squeeze
        
    def call(self, inputs):
        bs = tf.shape(inputs)[0]
        rows, cols = self.H // self.P, self.W // self.P
        patches = tf.reshape(inputs, [bs, rows, cols, -1, self.C])
        pats_by_clist = tf.unstack(patches, axis=-1)
        def tile_patches(ii):
            pats = tf.nn.embedding_lookup([pats_by_clist], int(ii))
            img = tf.nn.depth_to_space(pats, self.P)
            return img 
        img = tf.map_fn(fn=tile_patches, elems=tf.range(self.C), fn_output_signature=inputs.dtype)
        img = tf.squeeze(img, axis=-1)
        img = tf.transpose(img, perm=[1,2,3,0])
        C = tf.shape(img)[-1]
        img = tf.cond(tf.math.logical_and(tf.constant(self.is_squeeze), C==1), 
                      lambda: tf.squeeze(img), lambda: img)
        img=tf.reshape(img,(int(bs),int(self.H),int(self.W),int(self.C)))
        return img

class LearnableFourierPositionalEncoding(layers.Layer):
    def __init__(self,  F_dim: int, H_dim: int, D: int, gamma: float, **kwargs):
        super().__init__(**kwargs)
        """    
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

    def build(self,input_shape):
        _,self.N, self.G, self.M = input_shape
        self.Wr = layers.Dense( self.F_dim // 2, use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(self.gamma ** -2))
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp_1 = layers.Dense( self.H_dim,use_bias=True, activation=tf.nn.gelu)
        self.mlp_2=layers.Dense( self.D // self.G)

    def call(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        batch_size,N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = tf.math.cos(projected)
        sines = tf.math.sin(projected)

        K=(tf.concat([cosines, sines], axis=-1))
        F = 1 / np.sqrt(self.F_dim) * K
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp_1(F)
        Y = self.mlp_2(Y)

        # Step 3. Reshape to x's shape
        return Y

class FourierPositionalAttention(layers.Layer):
    def __init__(self, filters,patch_size, targeted_patchsize,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.patch_size=patch_size
        self.targeted_patchsize=targeted_patchsize
    def build(self,input_shape):
        self.shape=input_shape
        self.windowpatches=WindowPatches(self.patch_size)
        self.postionalEncode=LearnableFourierPositionalEncoding( self.filters, 1024, self.filters, 10)
        self.norm=tfa.layers.InstanceNormalization()
        self.patchestoimage=PatchesToImage(int(self.shape[1]),int(self.shape[2]),self.filters//16,self.patch_size)

    def call(self,mask):
        windows=self.windowpatches(mask)
        postionalencoded=self.postionalEncode(windows)
        encoded_img=self.norm(encoded_img)
        encoded_img=self.patchestoimage(encoded_img)
        encoded_img = tf.image.resize(encoded_img, (self.targeted_patchsize,self.targeted_patchsize), method="bilinear")

        return encoded_img
