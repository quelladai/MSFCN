import os
from datetime import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, AveragePooling2D, Conv2DTranspose, Add, \
    Cropping2D, ZeroPadding2D, Activation, Concatenate, Deconv2D, BatchNormalization, LeakyReLU, Lambda,AveragePooling2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.initializers import RandomNormal, Constant
from keras import losses
from keras import metrics
from keras import callbacks
import keras as k
import tensorflow as tf
import keras.backend as K
import glob
import math
import matplotlib.pyplot as plt
import cv2
# from tensorlayer.layers import *
# import gdal
from keras.layers import merge, Dropout, concatenate, add
from keras import regularizers
from layers import ConvOffset2D
from subpixel_conv2d import SubpixelConv2D

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def upsample_filt(shape):
    factor = (shape[0] + 1) // 2
    if shape[0] % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:shape[0], :shape[0]]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return filt.reshape(shape)

def expand_dim_backend(x):
    x1 = K.expand_dims(x, 0)
    #
    return x1

def hw_flatten(x):
    return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

def flatten(inputs):
    x, y = inputs
    return K.reshape(x, shape=[-1, K.int_shape(y)[1], K.int_shape(y)[2], K.int_shape(y)[3]])

def sflatten(x):
    return K.reshape(x, shape=[K.shape(x)[0], 64, 64, 4])

def squeeze_dim(input):
    output = K.squeeze(input, 2)
    return output

def keep(x):
    return x

def ASPP(input):
    conv1 = Conv2D(4, kernel_size=(1, 1), activation='relu', padding='same')(input)  # 64

    conv21 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(1, 1), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv22 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(2, 2), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv23 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(3, 3), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv24 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(4, 4), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv25 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(5, 5), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv26 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(6, 6), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conv27 = Conv2D(4, 3, activation='relu', strides=(1, 1), dilation_rate=(7, 7), padding='same',
                        kernel_initializer='he_normal')(conv1)
    conca2 = Concatenate(axis=-1)([conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv1])

    return conca2

# def transpose(list1):
#     return [list(row) for row in zip(*list1)]

# def bmm(x, y):#  x y 三维，bmm(x, y) 第一维度不变，只对后面两维度进行矩阵相乘
#     m_batchsize1, height1, width1 = x.size()
#     m_batchsize2, height2, width2 = y.size()
#     if m_batchsize1 != m_batchsize2:
#         print("can not calculate!")
#     x12 = np.zeros(height1, width1)
#     y12 = np.zeros(height2, width2)
#     xybmm = np.zeros(m_batchsize1, height1, width2)
#
#     for i in range (m_batchsize1):
#         for h1 in rannge(height1):
#             for w1 in range(width1):
#                 x12[h, w] = x[i, h1, w1]
#
#         for h2 in rannge(height2):
#             for w2 in range(width2):
#                 x12[h, w] = x[i, h2, w2]
#
#         xy12 = np.dot(x12, y12)
#
#         for h in range(height1):
#             for w in range(width2):
#                 xybmm[i, h, w] = xy12[h, w]
#
#     return xybmm

class myUnet(object):
    def __init__(self, img_rows = 256, img_cols = 256, weight_filepath=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = self.Dense_MSFCN_E()
        self.current_epoch = 0
        self.weight_filepath = weight_filepath
     # def CBRR_block(self, kn, ks, inputs):
    #
    #     conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
    #     conv1_bn = BatchNormalization()(conv1)
    #     conv1_relu = LeakyReLU(alpha=0)(conv1_bn)
    #     conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
    #     conv2_bn = BatchNormalization()(conv2)
    #     conv2_relu = LeakyReLU(alpha=0)(conv2_bn)
    #     merge = concatenate([inputs, conv2_relu], axis=3)
    #     return merge

    # def CBRR_block(self, kn, ks, inputs):
    #     conv_inputs = Conv2D(kn, ks, activation='relu', padding='same')(inputs)
    #
    #     conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
    #     conv1_bn = BatchNormalization()(conv1)
    #     conv1_relu = LeakyReLU(alpha=0)(conv1_bn)
    #
    #     conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
    #     conv2_bn = BatchNormalization()(conv2)
    #     conv2_relu = LeakyReLU(alpha=0)(conv2_bn)
    #
    #     # merge = Add()([conv_inputs, conv2_relu])
    #
    #     merge = Concatenate()([conv_inputs, conv2_relu])
    #
    #     return merge

    def sConv_block(self, kn, ks, inputs):
        conv = Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # pool = MaxPooling2D(pool_size=(2, 2))(conv)  # 256
        return conv

    def CBRR_block(self, kn, ks, inputs):
        # conv_inputs = Conv2D(kn, ks, activation='relu', padding='same')(inputs)

        # conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(kn, ks, activation='relu', padding='same')(inputs)

        conv2 = Conv2D(kn, ks, activation='relu', padding='same')(conv1)


        # conv1_relu = LeakyReLU(alpha=0)(conv1_bn)

        # merge = Add()([conv_inputs, conv1_relu])

        merge = Concatenate()([inputs, conv2])

        return merge

    def CBRR_block1(self, kn, ks, inputs):
        conv_inputs = Conv2D(kn, ks, activation='relu', padding='same')(inputs)

        conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
        # conv1_bn = BatchNormalization()(conv1)
        conv1_relu = LeakyReLU(alpha=0)(conv1)

        conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
        # conv2_bn = BatchNormalization()(conv2)
        conv2_relu = LeakyReLU(alpha=0)(conv2)

        merge = Add()([conv_inputs, conv2_relu])
        return merge

    def Res_MSFCN_ED(self):

        data = Input((self.img_rows, self.img_cols, 3))
        conv1 = BatchNormalization()(data)
        conv1 = LeakyReLU(alpha=0)(conv1)

        conv1 = self.CBRR_block(64, 3, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.CBRR_block(128, 3, pool1)
        conv2 = self.sConv_block(128, 3, conv2)  # 128
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.CBRR_block(256, 3, pool2)
        conv3 = self.sConv_block(256, 3, conv3)  # 64
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.CBRR_block(512, 3, pool3)
        conv4 = self.sConv_block(512, 3, conv4)  # 32
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.CBRR_block(1024, 3, pool4)
        conv5 = self.CBRR_block(1024, 3, conv5)  # 32

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        # conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        conv6 = self.CBRR_block(512, 3, merge6)
        conv6 = self.sConv_block(512, 3, conv6)  # 32
        b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        # conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        conv7 = self.CBRR_block(256, 3, merge7)
        conv7 = self.sConv_block(256, 3, conv7)  # 32
        b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        # conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        conv8 = self.CBRR_block(128, 3, merge8)
        conv8 = self.sConv_block(128, 3, conv8)  # 32
        b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        # conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        conv9 = self.CBRR_block(64, 3, merge9)
        conv9 = self.sConv_block(64, 3, conv9)  # 32
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        output = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(fuse)

        # output = Multiply()([output, data_3])
        #
        # model = Model(inputs=[data_1, data_2, data_3], outputs=output)
        model = Model(inputs=[data], outputs=output)

        # model = Model(inputs=[data_1, data_2], outputs=output)
        model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

        return model

    def Res_MSFCN_E(self):
        data = Input((self.img_rows, self.img_cols, 3))
        # conv1 = BatchNormalization()(data)
        # conv1 = LeakyReLU(alpha=0)(conv1)

        conv1 = self.CBRR_block(64, 3, data)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.CBRR_block(128, 3, pool1)
        # conv2 = self.sConv_block(128, 3, conv2)  # 128
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.CBRR_block(256, 3, pool2)
        # conv3 = self.sConv_block(256, 3, conv3)  # 64
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.CBRR_block(512, 3, pool3)
        # conv4 = self.sConv_block(512, 3, conv4)  # 32
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.CBRR_block(1024, 3, pool4)
        # conv5 = self.CBRR_block(1024, 3, conv5)  # 32

        # conv5 = dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        conv_channel = self.channel_attention(fuse)
        mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)
        # mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)

        # mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)
        model = Model(inputs=[data], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

    def Unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 256

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 128

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 64

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 32

        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(256, 2, strides=(2, 2), padding='same')(conv5))

        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(128, 2, strides=(2, 2), padding='same')(conv6))

        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(64, 2, strides=(2, 2), padding='same')(conv7))

        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(32, 2, strides=(2, 2), padding='same')(conv8))

        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        mask = Conv2D(3, 1, activation='softmax', padding='same')(conv9)
        model = Model(inputs=[inputs], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

    def MSUNet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
        conv6_ds = Conv2D(1, 1, activation='relu', padding='same')(conv6)
        b1 = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), activation=None, use_bias=False, padding='same',
                             name='b1', kernel_initializer=upsample_filt)(conv6_ds)

        up7 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)

        conv7_offset = ConvOffset2D(64, name='conv21_offset')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7_offset)

        conv7_offset = ConvOffset2D(64, name='conv21_offset')(conv7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7_offset)

        conv7_ds = Conv2D(1, 1, activation='relu', padding='same')(conv7)
        b2 = Conv2DTranspose(1, kernel_size=(8, 8), strides=(4, 4), activation=None, use_bias=False, padding='same',
                             name='b2', kernel_initializer=upsample_filt)(conv7_ds)

        up8 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)

        conv8_offset = ConvOffset2D(64, name='conv21_offset')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8_offset)

        conv8_offset = ConvOffset2D(64, name='conv21_offset')(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8_offset)

        # conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
        # conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
        conv8_ds = Conv2D(1, 1, activation='relu', padding='same')(conv8)
        b3 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation=None, use_bias=False, padding='same',
                             name='b3')(conv8_ds)

        up9 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)

        conv9_offset = ConvOffset2D(64, name='conv21_offset')(merge9)
        conv9 = Conv2D(128, 3, activation='relu', padding='same')(conv9_offset)

        conv9_offset = ConvOffset2D(64, name='conv21_offset')(conv9)
        conv9 = Conv2D(128, 3, activation='relu', padding='same')(conv9_offset)

        # conv9 = Conv2D(32, 3, activation='relu', padding='same')(merge9)
        # conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        fuse = Concatenate(axis=-1)([b1, b2, b3, b4])

        mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)
        model = Model(inputs=[inputs], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

    def fUnet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        conv6_ds = Conv2D(1, 1, activation='relu', padding='same')(conv6)
        b1 = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), activation=None, use_bias=False, padding='same',
                             name='b1', kernel_initializer=upsample_filt)(conv6_ds)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        conv7_ds = Conv2D(1, 1, activation='relu', padding='same')(conv7)
        b2 = Conv2DTranspose(1, kernel_size=(8, 8), strides=(4, 4), activation=None, use_bias=False, padding='same',
                             name='b2', kernel_initializer=upsample_filt)(conv7_ds)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        conv8_ds = Conv2D(1, 1, activation='relu', padding='same')(conv8)
        b3 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation=None, use_bias=False, padding='same',
                             name='b3')(conv8_ds)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        fuse = Concatenate(axis=-1)([b1, b2, b3, b4])

        mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)
        model = Model(inputs=[inputs], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

    def f_unet1(self):
        inputs = Input((self.img_rows, self.img_cols, 4))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        conv6_ds = Conv2D(1, 1, activation='relu', padding='same')(conv6)
        b1 = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), activation=None, use_bias=False, padding='same',
                             name='b1', kernel_initializer=upsample_filt)(conv6_ds)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        conv7_ds = Conv2D(1, 1, activation='relu', padding='same')(conv7)
        b2 = Conv2DTranspose(1, kernel_size=(8, 8), strides=(4, 4), activation=None, use_bias=False, padding='same',
                             name='b2', kernel_initializer=upsample_filt)(conv7_ds)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        conv8_ds = Conv2D(1, 1, activation='relu', padding='same')(conv8)
        b3 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), activation=None, use_bias=False, padding='same',
                             name='b3')(conv8_ds)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        fuse = Concatenate(axis=-1)([b1, b2, b3, b4])
        fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None,
                      kernel_initializer=Constant(value=0.2))(fuse)

        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

        model.compile(loss={'o1': "binary_crossentropy",
                            'o2': "binary_crossentropy",
                            'o3': "binary_crossentropy",
                            'o4': "binary_crossentropy",
                            'ofuse': "binary_crossentropy",
                            },
                      metrics={'ofuse': ofuse_pixel_error},
                      optimizer='adam')
        return model

    def f_unet2(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        b1 = Conv2D(1, 1, activation=None, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        b2 = Conv2D(1, 1, activation=None, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        b3 = Conv2D(1, 1, activation=None, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        b1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        b2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        b3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([b1, b2, b3, b4])
        fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None,
                      kernel_initializer=Constant(value=0.2))(fuse)

        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

        model.compile(loss={'o1': "binary_crossentropy",
                            'o2': "binary_crossentropy",
                            'o3': "binary_crossentropy",
                            'o4': "binary_crossentropy",
                            'ofuse': "binary_crossentropy",
                            },
                      metrics={'ofuse': ofuse_pixel_error},
                      optimizer='adam')
        return model

    def f_unet3(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])
        fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None,
                      kernel_initializer=Constant(value=0.2))(fuse)

        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

        model.compile(loss={'o1': "binary_crossentropy",
                            'o2': "binary_crossentropy",
                            'o3': "binary_crossentropy",
                            'o4': "binary_crossentropy",
                            'ofuse': "binary_crossentropy",
                            },
                      metrics={'ofuse': ofuse_pixel_error},
                      optimizer='adam')
        return model

    def res_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(drop5)
        merge6 = k.layers.concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv_sub4 = Conv2D(1, 1, activation='sigmoid')(conv6)

        conv_sub4_up = UpSampling2D(size=(2, 2))(conv_sub4)
        up7 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7_res = add([conv7, conv_sub4_up])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_res)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv_sub3 = Conv2D(1, 1, activation='sigmoid')(conv7)

        conv_sub3_up = UpSampling2D(size=(2, 2))(conv_sub3)
        up8 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8_res = add([conv8, conv_sub3_up])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_res)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv_sub2 = Conv2D(1, 1, activation='sigmoid')(conv8)

        conv_sub2_up = UpSampling2D(size=(2, 2))(conv_sub2)
        up9 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9_res = add([conv9, conv_sub2_up])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_res)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        # model = Model(inputs=inputs, outputs=conv10)
        model = Model(inputs=inputs, outputs=[conv10, conv_sub2, conv_sub3, conv_sub4])
        model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", loss_weights=[1, 0.5, 0.25, 0.125],
                      metrics=['accuracy'])
        return model

    # def CAM(self, x):
    #     m_batchsize, C, height, width = np.size()
    #     querry_conv = Conv2D(C, 1, padding='same')(x)
    #     key_conv = Conv2D(C, 1, padding='same')(x)
    #     value_conv = Conv2D(C, 1, padding='same')(x)
    #
    #     proj_querry = np.reshape(querry_conv, (m_batchsize, C, width * heigh))  # B C N
    #     proj_key = np.reshape(key_conv, (m_batchsize, C, width * heigh))  # B C N
    #     tproj_key = proj_key.transpose(0, 2, 1)  # B N C
    #     energy = K.batch_dot(proj_querry, tproj_key)  # B C C
    #
    #     attention = k.softmax(energy)
    #     proj_value = np.reshape(value_conv, (m_batchsize, C, width * heigh))  # B C N
    #     tattention = attention.transpose(0, 2, 1)  # B C C
    #     out = k.batch_dot(tattention, proj_value) # B C N
    #     out = np.reshape(out, (m_batchsize, C, height, width))
    #
    #     out = gamma * out + x
    #
    #     # out = Conv2D(C, 1)(out)
    #     # out = out + x
    #
    #     return out
    #
    # def PAM(self, x):
    #     m_batchsize, C, height, width = np.size(x)
    #     querry_conv = Conv2D(C, 1, padding='same')(x)
    #     key_conv = Conv2D(C, 1, padding='same')(x)
    #     value_conv = Conv2D(C, 1, padding='same')(x)
    #
    #     proj_querry = np.reshape(querry_conv, (m_batchsize, C, width * heigh)) # B C N
    #     tproj_querry = proj_querry.transpose(0, 2, 1) # B N C
    #     proj_key = np.reshape(key_conv, (m_batchsize, C, width * heigh))# B C N
    #     energy = K.batch_dot(tproj_querry, proj_key) # B N N
    #     attention = k.softmax(energy)
    #     proj_value = np.reshape(value_conv, (m_batchsize, C, width * heigh)) # B C N
    #     tattention = attention.transpose(0, 2, 1)  # B N N
    #     out = k.batch_dot(proj_value, tattention)
    #     out = np.reshape(out, (m_batchsize, C, height, width))
    #
    #     out = gamma * out + x
    #
    #     return out

    def hw_flatten(self, x, shape):
        return K.reshape(x, shape)

    def reshape_layer(self, x, shape):
        return Lambda(K.reshape)([x, shape])

    def batch_dot_layer(self, tensor):
        return Lambda(K.batch_dot)(tensor)

    def softmax_layer(self, tensor):
        return Lambda(K.softmax)(tensor)

    def permute_dimensions_layer(self):
        return Lambda(K.permute_dimensions)(tensor)

    def ol(self, x):
        return x[:,:,:,:]


    # def PAM(self, x):
    #     f = Conv2D(32, 1, padding='same')(x)  # [bs, h, w, c']
    #     g = Conv2D(32, 1, padding='same')(x)   # [bs, h, w, c']
    #     h = Conv2D(32, 1, padding='same')(x)  # [bs, h, w, c]
    #     # flatten_g = self.hw_flatten(g)   # [bs, N, c] N = h * w
    #     # flatten_f = self.hw_flatten(f)   # [bs, N, c]
    #
    #     flatten_g = K.reshape(g, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #     flatten_f = K.reshape(f, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #     flatten_h = K.reshape(h, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #
    #     # flatten_g = Lambda(lambda x: K.reshape(x,[1,512*512,4])(g))
    #     # flatten_f = Lambda(lambda x: K.reshape(x,[1,512*512,4])(f))
    #     # flatten_h = Lambda(lambda x: K.reshape(x,[1,512*512,4])(h))
    #
    #     s = K.batch_dot(flatten_g, K.permute_dimensions(flatten_f, (0, 2, 1)))  # [bs, N, N]
    #
    #     # s = Lambda(K.batch_dot)(flatten_g, Lambda(K.permute_dimensions)(flatten_f, (0, 2, 1)))   # [bs, N, N]
    #     # tf = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1))(flatten_f))
    #     # s = Lambda(lambda x: K.batch_dot(x)([flatten_g,tf]))
    #
    #     beta = K.softmax(s, axis=-1) # attention map   # [bs, N, N]
    #     # beta = Lambda(K.softmax)(s, axis=-1)     # attention map   # [bs, N, N]
    #     # attention map   # [bs, N, N]
    #     # flatten_h = self.hw_flatten(h) # [bs, N, c]
    #
    #     o = K.batch_dot(K.permute_dimensions(beta, (0, 2, 1)), flatten_h)  # [bs, N, C]
    #     o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
    #
    #     # o = Lambda(K.batch_dot)(Lambda(K.permute_dimensions)(beta, (0, 2, 1)), flatten_h)  # [bs, N, C]
    #     # o = Lambda(K.reshape)(o, shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]]) # [bs, h, w, C]
    #
    #     # gamma = tf.Variable(tf.zeros([1]), name="gamma")
    #     # #
    #     # x = Add()([gamma * o, x])
    #
    #     o = Conv2D(32, 1, padding='same')(o)
    #
    #     x = Add()([o, s])
    #     # x = Lambda(lambda x: squeeze(x, axis=[1, 2]))
    #     # x = Lambda(expand_dim_backend)(x)
    #     # x = Lambda(squeeze_dim)(x)
    #
    #     return x
    #
    #
    #
    # def CAM(self, x):
    #     f = Conv2D(32, 1, padding='same')(x)  # [bs, h, w, c']
    #     g = Conv2D(32, 1, padding='same')(x)  # [bs, h, w, c']
    #     h = Conv2D(32, 1, padding='same')(x)  # [bs, h, w, c]
    #
    #     flatten_g = K.reshape(g, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #     flatten_f = K.reshape(f, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #     flatten_h = K.reshape(h, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
    #
    #     # flatten_g = Lambda(K.reshape)(g, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])
    #     # flatten_f = Lambda(K.reshape)(f, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])
    #     # flatten_h = Lambda(K.reshape)(h, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])
    #
    #     # flatten_g = self.hw_flatten(g)
    #     # flatten_f = self.hw_flatten(f)
    #     s = K.batch_dot(K.permute_dimensions(flatten_g, (0, 2, 1)), flatten_f) # [bs, c, c]
    #
    #     # s = Lambda(K.batch_dot)(Lambda(K.permute_dimensions)(flatten_g, (0, 2, 1)), flatten_f)  # [bs, c, c]
    #
    #     beta = K.softmax(s, axis=-1)   # attention map
    #
    #     # beta =Lambda(K.softmax)(s, axis=-1)  # attention map
    #
    #     o = K.batch_dot(flatten_h, K.permute_dimensions(beta, (0, 2, 1)))  # [bs, N, C]
    #     o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
    #     # o = Lambda(self.ol)(o)
    #
    #     # o = Lambda(K.batch_dot)(flatten_h, Lambda(K.permute_dimensions)(beta, (0, 2, 1)))  # [bs, N, C]
    #     # o = Lambda(K.reshape)(o, shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]])  # [bs, h, w, C]
    #
    #     # gamma = tf.Variable(tf.zeros([1]), name="gamma")
    #     #
    #     # x = Add()([gamma * o, x])
    #     o = Conv2D(4, 1, padding='same')(o)
    #
    #     x = Add()([o, x])
    #     # x = Lambda(lambda x: squeeze(x, axis=[1, 2]))
    #     # x = Lambda(expand_dim_backend)(x)
    #     # x = Lambda(squeeze_dim)(x)
    #
    #     return x

    def spatial_attention(self,  x,  scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):

            sf = Conv2D(int(K.int_shape(x)[3]/32), 1, padding='same')(x)  # [bs, h, w, c']
            sg = Conv2D(int(K.int_shape(x)[3]/32), 1, padding='same')(x)  # [bs, h, w, c']
            sh = Conv2D(int(K.int_shape(x)[3]/32), 1, padding='same')(x)  # [bs, h, w, c]

            ss = Lambda(self.mulbt)([Lambda(hw_flatten)(sg), Lambda(hw_flatten)(sf)])  # [bs, n , n]

            sbeta = Lambda(tf.nn.softmax)(ss)  # attention map [bs, n , n]

            so = Lambda(self.mulat)([sbeta, Lambda(hw_flatten)(sh)])  # [bs, N, C]

            so_reshape = Lambda(flatten)([so, sf])

            cchange = Conv2D(int(K.int_shape(x)[3]), 1, padding='same')(so_reshape)
            conv_channel = Add()([cchange, x])

        return conv_channel

    def channel_attention(self, x, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # f, g, h = inputs

            # f = Conv2D(4, 1, padding='same')(x)  # [bs, h, w, c']
            # g = Conv2D(4, 1, padding='same')(x)  # [bs, h, w, c']
            # h = Conv2D(4, 1, padding='same')(x)  # [bs, h, w, c]
            #
            # # N = h * w
            # s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_a=True)  # [bs, c , c]
            # # s = Lambda(tf.matmul, arguments={'transpose_a':True})([hw_flatten(g), hw_flatten(f)])
            #
            # beta = tf.nn.softmax(s)  # attention map [bs, c , c]
            #
            # o = tf.matmul(hw_flatten(h), beta, transpose_b=True)  # [bs, N, C]
            # gamma = tf.Variable(tf.zeros([1]), name="gammac")
            #
            # o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
            # x = gamma * o + x

            cf = Conv2D(int(K.int_shape(x)[3]), 1, padding='same')(x)  # [bs, h, w, c']
            cg = Conv2D(int(K.int_shape(x)[3]), 1, padding='same')(x)  # [bs, h, w, c']
            ch = Conv2D(int(K.int_shape(x)[3]), 1, padding='same')(x)  # [bs, h, w, c]

            cs = Lambda(self.mulat)([Lambda(hw_flatten)(cg), Lambda(hw_flatten)(cf)])  # [bs, c , c]

            cbeta = Lambda(tf.nn.softmax)(cs)  # attention map [bs, c , c]

            co = Lambda(self.mulbt)([Lambda(hw_flatten)(ch), cbeta])  # [bs, N, C]

            co_reshape = Lambda(flatten)([co, cf])

            cchange = Conv2D(int(K.int_shape(x)[3]), 1, padding='same')(co_reshape)
            conv_channel = Add()([cchange, x])

        return conv_channel

    def multiply(self, inputs):
        x, y = inputs
        return x * y

    def mulat(self, inputs):
        x, y = inputs
        return tf.matmul(x, y, transpose_a=True)

    def mulbt(self, inputs):
        x, y = inputs
        return tf.matmul(x, y, transpose_b=True)

    def MSFCN(self):
        data = Input((self.img_rows, self.img_cols, 3)) # 512

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(data)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        # drop1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 256
        # conv_channel1 = self.channel_attention(pool1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        # drop2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 128
        # conv_channel2 = self.channel_attention(pool2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        # drop3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 64
        # conv_channel3 = self.channel_attention(pool3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        # drop4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 32
        # conv_channel4 = self.channel_attention(pool4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5) #32
        # drop5 = BatchNormalization()(conv5)
        # conv_spatial = self.spatial_attention(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        # conv6_offset = ConvOffset2D(512, name='conv61_offset')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)

        # conv6_offset = ConvOffset2D(512, name='conv62_offset')(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        # conv_channel6 = self.channel_attention(conv6)
        # drop6 = BatchNormalization()(conv6)
        b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)

        # conv7_offset = ConvOffset2D(256, name='conv71_offset')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)

        # conv7_offset = ConvOffset2D(256, name='conv72_offset')(conv7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7_offset)
        #
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        # conv_channel7 = self.channel_attention(conv7)
        # drop7 = BatchNormalization()(conv7)
        b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)

        # conv8_offset = ConvOffset2D(128, name='conv81_offset')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)

        # conv8_offset = ConvOffset2D(128, name='conv82_offset')(conv8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8_offset)
        #
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        # conv_channel8 = self.channel_attention(conv8)
        # drop8 = BatchNormalization()(conv8)
        b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)

        # conv9_offset = ConvOffset2D(64, name='conv91_offset')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)

        # conv9_offset = ConvOffset2D(64, name='conv92_offset')(conv9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9_offset)

        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        # conv_channel9 = self.channel_attention(conv9)
        # # drop9 = BatchNormalization()(conv9)
        b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        # dfuse = MaxPooling2D(pool_size=(8, 8))(fuse) # 128
        # dconv_spatial = self.spatial_attention(dfuse)
        # conv_spatial = UpSampling2D(size=(8, 8), data_format=None)(dconv_spatial)

        # conv_channel_input = Conv2D(4, 1, activation='relu', padding='same')(fuse)

        # conv_channel = self.channel_attention(fuse)
        # # # feature_sum = Add()([conv_spatial, conv_channel])
        # #
        # #
        # # # cf = Conv2D(4, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # cg = Conv2D(4, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # ch = Conv2D(4, 1, padding='same')(fuse)  # [bs, h, w, c]
        # # # #
        # # # # conv_channel = Lambda(self.channel_attention)([cf, cg, ch]) # # [bs, c, c]
        # # #
        # # # # cs = tf.matmul(hw_flatten(cg), hw_flatten(cf), transpose_a=True)  # [bs, c , c]
        # # # cs = Lambda(self.mulat)([Lambda(hw_flatten)(cg), Lambda(hw_flatten)(cf)])  # [bs, c , c]
        # # #
        # # # cbeta = Lambda(tf.nn.softmax)(cs)  # attention map [bs, c , c]
        # # #
        # # # # co = tf.matmul(hw_flatten(ch), cbeta, transpose_b=True)  # [bs, N, C]
        # # # co = Lambda(self.mulbt)([Lambda(hw_flatten)(ch), cbeta])  # [bs, N, C]
        # # #
        # # # co_reshape = Lambda(flatten)(co)
        # # # # cgamma = tf.Variable(tf.zeros([1]), name="gammac")
        # # # # cgamma = 0.5
        # # # # cchange = Lambda(self.multiply)([0.5, Lambda(flatten)(co)])
        # # # # cchange = cgamma * co_reshape
        # # # #
        # # # # # conv_channel = K.reshape(conv_channel, shape=K.shape(fuse))  # [bs, h, w, C]
        # # # # # conv_channel = cgamma * conv_channel + fuse
        # # # # conv_channel = Add()([cf, fuse])
        # # #
        # # # cchange = Conv2D(4, 1, padding='same')(co_reshape)
        # # # conv_channel = Add()([cchange, fuse])
        # #
        # #
        # # # N = h * w
        # # # fcg = hw_flatten(cg)
        # #
        # # # cs = tf.matmul(hw_flatten(cg), hw_flatten(cf), transpose_a=True)  # [bs, c , c]
        # # #
        # # # cbeta = tf.nn.softmax(cs)  # attention map [bs, c , c]
        # # # #
        # # # co = tf.matmul(hw_flatten(ch), cbeta, transpose_b=True)  # [bs, N, C]
        # # # cgamma = tf.Variable(tf.zeros([1]), name="gammac")
        # # #
        # # # # co = K.reshape(co, shape=K.shape(fuse))  # [bs, h, w, C]
        # # # co = Lambda(flatten)(co)
        # # # # cochange = cgamma * co
        # # # # channel_conv = Add()([cgamma * co, fuse])
        # # # # channel_conv = flatten(channel_conv, fuse)
        # # # # co = Lambda(lambda co: K.reshape(co, shape=K.shape(fuse)))
        # # # # cochange = cgamma * co
        # # # # # channel_conv = cgamma * co + fuse
        # # # # channel_conv = Lambda(self.add)([cochange, fuse])
        # # #
        # # # # channel_conv = Lambda(lambda channel_conv: tf.image.resize_images(channel_conv, [512, 512], method=0))(
        # # # #     channel_conv)
        # # #
        # # # # poolfuse = MaxPooling2D(pool_size=(8, 8))(fuse)  # 128
        # #
        # # # # channel attention
        # # #
        # # # cf = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # cg = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # ch = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c]
        # # #
        # # # # flatten_cf = K.reshape(cf, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # # # flatten_cg = K.reshape(cg, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # # # flatten_ch = K.reshape(ch, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # #
        # # # flatten_cg = Lambda(lambda flatten_cg: tf.image.resize_images(flatten_cg, [512 * 512, 1], method=0))(cg)
        # # # flatten_cg = K.squeeze(flatten_cg, 2)
        # # # flatten_cf = Lambda(lambda flatten_cf: tf.image.resize_images(flatten_cf, [512 * 512, 1], method=0))(cf)
        # # # flatten_cf = K.squeeze(flatten_cf, 2)
        # # # flatten_ch = Lambda(lambda flatten_ch: tf.image.resize_images(flatten_ch, [512 * 512, 1], method=0))(ch)
        # # # flatten_ch = K.squeeze(flatten_ch, 2)
        # # #
        # # # cs = K.batch_dot(K.permute_dimensions(flatten_cg, (0, 2, 1)), flatten_cf)  # [bs, c, c]
        # # #
        # # # cbeta = K.softmax(cs, axis=-1)  # attention map
        # # #
        # # # cotbetah = K.batch_dot(flatten_ch, K.permute_dimensions(cbeta, (0, 2, 1)))  # [bs, N, C]
        # # # coreshape = K.reshape(cotbetah, shape=K.shape(fuse))  # [bs, h, w, C]
        # # #
        # # # # co = Conv2D(4, 1, padding='same')(coreshape)
        # # # #
        # # # # channel_conv = Add()([co, fuse])
        # # # gamma = tf.Variable(tf.zeros([1]), name="gamma")
        # # #
        # # # channel_conv = Add()([gamma * coreshape, fuse])
        # # #
        # # # #spatial attention
        # # #
        # # # sf = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # sg = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c']
        # # # sh = Conv2D(32, 1, padding='same')(fuse)  # [bs, h, w, c]
        # # #
        # # # flatten_sg = Lambda(lambda flatten_sg: tf.image.resize_images(flatten_sg, [512 * 512, 1], method=0))(sg)
        # # # flatten_sg = K.squeeze(flatten_sg, 2)
        # # # flatten_sf = Lambda(lambda flatten_sf: tf.image.resize_images(flatten_sf, [512 * 512, 1], method=0))(sf)
        # # # flatten_sf = K.squeeze(flatten_sf, 2)
        # # # flatten_sh = Lambda(lambda flatten_sh: tf.image.resize_images(flatten_sh, [512 * 512, 1], method=0))(sh)
        # # # flatten_sh = K.squeeze(flatten_sh, 2)
        # # #
        # # # # flatten_sf = K.reshape(sf, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # # # flatten_sg = K.reshape(sg, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # # # flatten_sh = K.reshape(sh, shape=[K.shape(fuse)[0], K.shape(fuse)[1] * K.shape(fuse)[2], K.shape(fuse)[3]])
        # # #
        # # # ss = K.batch_dot(flatten_sg, K.permute_dimensions(flatten_sf, (0, 2, 1)))  # [bs, N, N]
        # # #
        # # # sbeta = K.softmax(ss, axis=-1)  # attention map   # [bs, N, N]
        # # #
        # # # sohtbeat = K.batch_dot(K.permute_dimensions(sbeta, (0, 2, 1)), flatten_sh)  # [bs, N, C]
        # # # soreshape = K.reshape(sohtbeat, shape=K.shape(fuse))  # [bs, h, w, C]
        # # #
        # # # # so = Conv2D(4, 1, padding='same')(soreshape)
        # # # #
        # # # # spatial_conv = Add()([so, fuse])
        # # #
        # # # gamma = tf.Variable(tf.zeros([1]), name="gamma")
        # # #
        # # # # spatial_conv = Add()([gamma * soreshape, fuse])
        # # # spatial_conv = Add()([gamma * soreshape, fuse])
        # #
        # # # spatial_conv =self.PAM(fuse)
        # # # channel_conv = self.CAM(fuse)
        # # # spatial_conv = Lambda(lambda spatial_conv: tf.image.resize_images(spatial_conv, [512, 512], method=0))(spatial_conv)
        # # # channel_conv = Lambda(lambda channel_conv: tf.image.resize_images(channel_conv, [512, 512], method=0))(channel_conv)
        # #
        # # # feature_sum = self.sum_layer([spatial_conv, channel_conv])
        # #
        # # # spatial_conv = Lambda(self.spatial_attention)(poolfuse)
        # # # channel_conv = self.channel_attention(fuse)
        # #
        # # # channel_conv = Lambda(self.channel_attention)(fuse)
        # #
        # # # feature_sum = Add()([spatial_conv, channel_conv])
        # # # upfeature_sum = Conv2DTranspose(4, kernel_size=(1, 1), strides=(8, 8), activation='relu', padding='same')(feature_sum)
        # #
        # # # conv_channel = self.channel_attention(fuse)
        # #
        # conv_channel = self.channel_attention(fuse)
        # # mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)
        # mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)
        mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)

        model = Model(inputs=[data], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model
        #

    def Dense_MSFCN_E(self):
        data = Input((self.img_rows, self.img_cols, 3))  # 512
        # data = Add()([conv_1, conv_2])",
        numfilter = 256

        conv1 = Conv2D(numfilter, 3, activation='relu', padding='same')(data)
        conv1 = Conv2D(numfilter, 3, activation='relu', padding='same')(conv1)
        conv1 = Concatenate(axis=-1)([data, conv1]) # 3n
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(numfilter, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(numfilter, 3, activation='relu', padding='same')(conv2)
        conv2 = Concatenate(axis=-1)([pool1, conv2]) #4n
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(numfilter, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(numfilter, 3, activation='relu', padding='same')(conv3)
        conv3 = Concatenate(axis=-1)([pool2, conv3]) #5n
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(numfilter, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(numfilter, 3, activation='relu', padding='same')(conv4)
        conv4 = Concatenate(axis=-1)([pool3, conv4]) #6n
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(numfilter, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(numfilter, 3, activation='relu', padding='same')(conv5)
        conv5 = Concatenate(axis=-1)([pool4, conv5]) #7n

        #multi-scale
        # AvPool_1 = AveragePooling2D(pool_size=(16, 16))(conv5)
        # conv_11 = Conv2D(256, 1, padding='same')(AvPool_1)
        # B_1 = UpSampling2D(size=(16, 16), data_format=None, interpolation='bilinear')(conv_11)
        # # B_1 = SubpixelConv2D(upsampling_factor=16)(conv_11)
        # conv_12 = Conv2D(256, 3, padding='same')(B_1)
        #
        # relu_2 = K.activations.relu()(conv_12)
        # AvPool_2 = AveragePooling2D(pool_size=(8, 8))(relu_2)
        # conv_21 = Conv2D(256, 1, padding='same')(AvPool_2)
        # B_2 = UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear')(conv_21)
        # # B_2 = SubpixelConv2D(upsampling_factor=16)(conv_21)
        # conv_22 = Conv2D(256, 3, padding='same')(B_2)
        #
        # relu_3 = K.activations.relu()(conv_22)
        # AvPool_3 = AveragePooling2D(pool_size=(4, 4))(relu_3)
        # conv_31 = Conv2D(256, 1, padding='same')(AvPool_3)
        # B_3 = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(conv_31)
        # # B_3 = SubpixelConv2D(upsampling_factor=16)(conv_31)
        # conv_32 = Conv2D(256, 3, padding='same')(B_3)
        #
        # relu_4 = K.activations.relu()(conv_32)
        # AvPool_4 = AveragePooling2D(pool_size=(2, 2))(relu_4)
        # conv_41 = Conv2D(256, 1, padding='same')(AvPool_4)
        # B_4 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv_41)
        # # B_4 = SubpixelConv2D(upsampling_factor=16)(conv_41)
        # conv_42 = Conv2D(256, 3, padding='same')(B_4)
        #
        # concate4 = Concatenate(axis=-1)([conv_12, conv_22, conv_32, conv_42])  # 1024

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        merge6 = k.layers.concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        b1 = Conv2D(3, 1, activation=None, use_bias=False, padding='same')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        b2 = Conv2D(3, 1, activation=None, use_bias=False, padding='same')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        b3 = Conv2D(3, 1, activation=None, use_bias=False, padding='same')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        b4 = Conv2D(3, 1, activation=None, use_bias=False, padding='same', name='b4')(conv9)

        ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)
        # ob1 = SubpixelConv2D(upsampling_factor=8)(b1)
        # ob2 = SubpixelConv2D(upsampling_factor=4)(b2)
        # ob3 = SubpixelConv2D(upsampling_factor=2)(b3)

        fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        # up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(conv5)
        # merge6 = k.layers.concatenate([conv4, up6], axis=3)
        # # conv6_offset = ConvOffset2D(512, name='conv61_offset')(merge6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        #
        # conv6_offset = ConvOffset2D(512, name='conv62_offset')(conv6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6_offset)
        # # conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        # # conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)
        # # conv_channel6 = self.channel_attention(conv6)
        # # drop6 = BatchNormalization()(conv6)
        # b1 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv6)
        #
        # up7 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv6)
        # merge7 = k.layers.concatenate([conv3, up7], axis=3)
        #
        # # conv7_offset = ConvOffset2D(256, name='conv71_offset')(merge7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        #
        # conv7_offset = ConvOffset2D(256, name='conv72_offset')(conv7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7_offset)
        # #
        # # conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        # # conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
        # # conv_channel7 = self.channel_attention(conv7)
        # # drop7 = BatchNormalization()(conv7)
        # b2 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv7)
        #
        # up8 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv7)
        # merge8 = k.layers.concatenate([conv2, up8], axis=3)
        #
        # # conv8_offset = ConvOffset2D(128, name='conv81_offset')(merge8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        #
        # conv8_offset = ConvOffset2D(128, name='conv82_offset')(conv8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8_offset)
        # #
        # # conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        # # conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
        # # conv_channel8 = self.channel_attention(conv8)
        # # drop8 = BatchNormalization()(conv8)
        # b3 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv8)
        #
        # up9 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(conv8)
        # merge9 = k.layers.concatenate([conv1, up9], axis=3)
        #
        # # conv9_offset = ConvOffset2D(64, name='conv91_offset')(merge9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        #
        # conv9_offset = ConvOffset2D(64, name='conv92_offset')(conv9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9_offset)
        #
        # # conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        # # conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        # # conv_channel9 = self.channel_attention(conv9)
        # # # drop9 = BatchNormalization()(conv9)
        # b4 = Conv2D(1, 1, activation=None, use_bias=False, padding='same')(conv9)
        #
        # ob1 = UpSampling2D(size=(8, 8), data_format=None)(b1)
        # ob2 = UpSampling2D(size=(4, 4), data_format=None)(b2)
        # ob3 = UpSampling2D(size=(2, 2), data_format=None)(b3)
        #
        # fuse = Concatenate(axis=-1)([ob1, ob2, ob3, b4])

        conv_channel = self.channel_attention(fuse)
        #
        # # BRF
        # # mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)
        # # residual = Conv2D(3, 3, activation='relu', padding='same')(mask)
        # # residual = Conv2D(3, 3, padding='same')(mask)
        # # mask_out = Add()([mask, residual])
        # #
        #
        mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)
        #
        # mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)  ##二分类用sigmoid 多分类用softmax'

        model = Model(inputs=[data], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-4),
                      # loss="binary_crossentropy",
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])

        # model.compile(optimizer=Adam(lr=1e-3, momentum=0.9, weight_decay=1e-5),loss="categorical_crossentropy",
        #               metrics=['accuracy'])

        return model

    def cloudNet(self):
        # RSE based on FCN+ASPP
        data = Input((self.img_rows, self.img_cols, 3))  # 128

        ASPP1 = ASPP(data)
        ASPP2 = ASPP(ASPP1)
        ASPP3 = ASPP(ASPP2)
        ASPP4 = ASPP(ASPP3)
        ASPP5 = ASPP(ASPP4)
        ASPP6 = ASPP(ASPP5)
        ASPP7 = ASPP(ASPP6)
        ASPP8 = ASPP(ASPP7)
        ASPP9 = ASPP(ASPP8)

        Conv_out = Conv2D(3, 1, activation='softmax', padding='same')(ASPP9)
        model = Model(inputs=[data], outputs=[Conv_out])
        model.compile(optimizer=Adam(lr=1e-2, momentum=0.95, weight_decay=0.00005),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

    def cloudNetshao(self):

        #邵振峰 TGRS

        data = Input((self.img_rows, self.img_cols, 3))  # 128
        conv1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(data) #64
        conv2 = Conv2D(96, 3, padding='same')(conv1)
        bn2 = BatchNormalization()(conv2)
        relu2 = k.layers.Activation('relu')(bn2)
        conv3 = Conv2D(128, 3, padding='same')(relu2)
        bn3 = BatchNormalization()(conv3)
        relu3 = k.layers.Activation('relu')(bn3)  #64
        pool3 = MaxPooling2D(pool_size=(2, 2))(relu3) #32

        conv4 = Conv2D(196, 3, padding='same')(pool3)
        bn4 = BatchNormalization()(conv4)
        relu4 = k.layers.Activation('relu')(bn4)  # 32
        conv5 = Conv2D(256, 3, padding='same')(relu4)
        bn5 = BatchNormalization()(conv5)
        relu5 = k.layers.Activation('relu')(bn5)  # 32
        pool5 = MaxPooling2D(pool_size=(2, 2))(relu5)  # 16

        conv6 = Conv2D(256, 3, padding='same')(pool5)
        bn6 = BatchNormalization()(conv6)
        relu6 = k.layers.Activation('relu')(bn6)  # 16
        conv7 = Conv2D(512, 3, padding='same')(relu6)
        bn7 = BatchNormalization()(conv7)
        relu7 = k.layers.Activation('relu')(bn7)  # 16

        AvPool_1 = AveragePooling2D(pool_size=(16, 16))(relu7)
        conv_11 = Conv2D(256, 1, padding='same')(AvPool_1)
        B_1 = UpSampling2D(size=(16, 16), data_format=None, interpolation='bilinear')(conv_11)
        conv_12 = Conv2D(256, 3, padding='same')(B_1)

        relu_2 = k.layers.Activation('relu')(conv_12)
        AvPool_2 = AveragePooling2D(pool_size=(8, 8))(relu_2)
        conv_21 = Conv2D(256, 1, padding='same')(AvPool_2)
        B_2 = UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear')(conv_21)
        conv_22 = Conv2D(256, 3, padding='same')(B_2)

        relu_3 = k.layers.Activation('relu')(conv_22)
        AvPool_3 = AveragePooling2D(pool_size=(4, 4))(relu_3)
        conv_31 = Conv2D(256, 1, padding='same')(AvPool_3)
        B_3 = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(conv_31)
        conv_32 = Conv2D(256, 3, padding='same')(B_3)

        relu_4 = k.layers.Activation('relu')(conv_32)
        AvPool_4 = AveragePooling2D(pool_size=(2, 2))(relu_4)
        conv_41 = Conv2D(256, 1, padding='same')(AvPool_4)
        B_4 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv_41)
        conv_42 = Conv2D(256, 3, padding='same')(B_4)

        concate4 = Concatenate(axis=-1)([conv_12, conv_22, conv_32, conv_42])  # 1024

        concate41 = Concatenate(axis=-1)([concate4, relu7])

        conv_up1 = Conv2D(512, 3, padding='same')(concate41)
        relu_up1 = k.layers.Activation('relu')(conv_up1)
        UpSampling_1 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(relu_up1)
        con_1 = Concatenate(axis=-1)([UpSampling_1, relu5])
        conv_up2 = Conv2D(256, 3, padding='same')(con_1)
        bn_2 = BatchNormalization()(conv_up2)
        relu_up2 = k.layers.Activation('relu')(bn_2)
        UpSampling_2 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(relu_up2)
        con_2 = Concatenate(axis=-1)([UpSampling_2, relu3])
        conv_up3 = Conv2D(128, 3, padding='same')(con_2)
        bn_3 = BatchNormalization()(conv_up3)
        relu_up3 = k.layers.Activation('relu')(bn_3)
        UpSampling_3 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(relu_up3)
        drop = Dropout(0.5)(UpSampling_3)

        Conv_out = Conv2D(3, 1, activation='softmax', padding='same')(drop)

        # model = Model(inputs=[data], outputs=[Conv_out])
        model = Model(inputs=[data], outputs=[Conv_out])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model

        # model.compile(optimizer=Adam(lr=1e-3, momentum=0.9, weight_decay=1e-5),
        #           loss="categorical_crossentropy",
        #           metrics=['accuracy'])
        # return model


        #RSE based on FCN

    def cloud_shadow_Net(self):
        data = Input((self.img_rows, self.img_cols, 3))  # 128
        conv1 = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(data)  # 64
        conv1 = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv1)  # 64
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 32

        conv2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool1)  # 64
        conv2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv2)  # 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 32

        conv3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool2)  # 64
        conv3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)  # 64
        conv3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)  # 64
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool3)  # 64
        conv4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)  # 64
        conv4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)  # 64
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 32

        conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool4)  # 64
        conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv5)  # 64
        conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv5)  # 64
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)  # 32

        up6 = UpSampling2D(size=(1, 1), data_format=None, interpolation='bilinear')(pool5)
        merge6 = k.layers.concatenate([pool5, up6], axis=3)
        conv6 = Conv2DTranspose(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2DTranspose(512, 3, activation='relu', padding='same')(conv6)
        conv6 = Conv2DTranspose(512, 3, activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv6)
        merge7 = k.layers.concatenate([pool4, up7], axis=3)
        conv7 = Conv2DTranspose(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2DTranspose(256, 3, activation='relu', padding='same')(conv7)
        conv7 = Conv2DTranspose(256, 3, activation='relu', padding='same')(conv7)

        up8 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv7)
        merge8 = k.layers.concatenate([pool3, up8], axis=3)
        conv8 = Conv2DTranspose(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2DTranspose(128, 3, activation='relu', padding='same')(conv8)
        conv8 = Conv2DTranspose(128, 3, activation='relu', padding='same')(conv8)

        up9 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv8)
        merge9 = k.layers.concatenate([pool2, up9], axis=3)
        conv9 = Conv2DTranspose(512, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2DTranspose(512, 3, activation='relu', padding='same')(conv9)

        up10 = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(conv9)
        merge10 = k.layers.concatenate([pool1, up10], axis=3)
        conv10 = Conv2DTranspose(512, 3, activation='relu', padding='same')(merge10)
        conv10 = Conv2DTranspose(512, 3, activation='relu', padding='same')(conv10)

    def CBRR_block(self, kn, ks, inputs):
        # conv_inputs =  Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
        # conv1_bn = BatchNormalization()(conv1)
        conv1_relu = LeakyReLU(alpha=0)(conv1)
        # merge = concatenate([inputs, conv_relu], axis=3)
        conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
        # conv2_bn = BatchNormalization()(conv2)
        conv2_relu = LeakyReLU(alpha=0)(conv2)
        conv3 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv2_relu)
        # conv3_bn = BatchNormalization()(conv3)
        conv3_relu = LeakyReLU(alpha=0)(conv3)
        merge = Add()([inputs, conv3_relu])
        # merge = concatenate([inputs, conv3_relu], axis=3)
        return merge

    def sConv_block(self, kn, ks, inputs):
        conv = Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)  # 256

        return pool

    def MSCN(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0)(conv1)
        conv2 = self.CBRR_block(32, 3, conv1)
        conv3 = self.sConv_block(32, 3, conv2)  # 256
        conv4 = self.CBRR_block(32, 3, conv3)
        conv5 = self.sConv_block(32, 3, conv4)  # 128
        conv6 = self.CBRR_block(32, 3, conv5)
        conv7 = self.sConv_block(32, 3, conv6)  # 64
        conv8 = self.CBRR_block(32, 3, conv7)
        conv9 = self.sConv_block(32, 3, conv8)  # 32
        conv10 = self.CBRR_block(32, 3, conv9)
        conv11 = self.sConv_block(32, 3, conv10)  # 16

        conv12 = Deconv2D(32, 2, strides=(2, 2), padding='same')(conv11) # 32
        conv13 = self.CBRR_block(32, 2, conv12)
        merge13 = concatenate([conv13, conv10], axis=3)

        conv14 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge13) # 64
        conv15 = self.CBRR_block(32, 2, conv14)
        merge15 = concatenate([conv15, conv8], axis=3)

        conv16 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge15)  # 128
        conv17 = self.CBRR_block(32, 2, conv16)
        merge17 = concatenate([conv17, conv6], axis=3)

        conv18 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge17)  # 256
        conv19 = self.CBRR_block(32, 2, conv18)
        merge19 = concatenate([conv19, conv4], axis=3)

        conv20 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge19)  # 512
        conv21 = self.CBRR_block(32, 2, conv20)
        merge21 = concatenate([conv21, conv2], axis=3)

        b1_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv11)
        b1_bn = BatchNormalization()(b1_conv)
        b1_relu = LeakyReLU(alpha=0)(b1_bn)
        b1 = UpSampling2D(size=(32, 32), data_format=None)(b1_relu)

        b2_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge13)
        b2_bn = BatchNormalization()(b2_conv)
        b2_relu = LeakyReLU(alpha=0)(b2_bn)
        b2 = UpSampling2D(size=(16, 16), data_format=None)(b2_relu)

        b3_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge15)
        b3_bn = BatchNormalization()(b3_conv)
        b3_relu = LeakyReLU(alpha=0)(b3_bn)
        b3 = UpSampling2D(size=(8, 8), data_format=None)(b3_relu)

        b4_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge17)
        b4_bn = BatchNormalization()(b4_conv)
        b4_relu = LeakyReLU(alpha=0)(b4_bn)
        b4 = UpSampling2D(size=(4, 4), data_format=None)(b4_relu)

        b5_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge19)
        b5_bn = BatchNormalization()(b5_conv)
        b5_relu = LeakyReLU(alpha=0)(b5_bn)
        b5 = UpSampling2D(size=(2, 2), data_format=None)(b5_relu)

        fuse = concatenate([b1, b2, b3, b4, b5, merge21], axis=3)

        mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)

        model = Model(inputs=[inputs], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-6),
                      # loss="binary_crossentropy",
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model
        # mask = Conv2D(3, 1, activation='softmax', padding='same')(conv_channel)

        # mask = Conv2D(3, 1, activation='sigmoid', padding='same')(fuse)  ##二分类用sigmoid 多分类用softmax'

        # model = Model(inputs=[data], outputs=[mask])
        #
        # model.compile(optimizer=Adam(lr=1e-4),
        #               # loss="binary_crossentropy",
        #               loss="categorical_crossentropy",
        #               metrics=['accuracy'])

        # mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)
        #
        # model = Model(inputs=[inputs], outputs=[mask])
        #
        # model.compile(optimizer=Adam(lr=1e-4),
        #               loss="categorical_crossentropy",
        #               metrics=['accuracy'])
        # return model


    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        # Loop over epochs
        for _ in range(epochs):

            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()

    def predict(self, sample):
        return self.model.predict(sample)

    def summary(self):
        print(self.model.summary())

    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath):
        self.model = self.Dense_MSFCN_E()

        epoch = int(os.path.basename(filepath).split("_")[0])
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')



