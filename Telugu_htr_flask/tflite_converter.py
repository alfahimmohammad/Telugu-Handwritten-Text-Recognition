# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:35:32 2020

@author: Kather
"""
#%%
import numpy as np
import tensorflow as tf
import keras.layers #install keras==2.2.4 and tensorflow==1.13.1
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import average, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM

unicodes = list(np.load('unicodes.npy',allow_pickle=True))

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def res_block(inputs,
              num_filters=16,
              kernel_size=3,
              strides=1,
              activation='relu',
              batch_normalization=True,
              conv_first=True,
              BN=True,
              A=True):
    x = inputs
    y = resnet_layer(inputs=x,
                    num_filters=num_filters,
                    strides=strides)
    y = resnet_layer(inputs=y,
                    num_filters=num_filters,
                    activation=None)
    x = resnet_layer(inputs=x,
                    num_filters=num_filters,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
    x = keras.layers.add([x, y])
    if BN:
      x = BatchNormalization()(x)
    if A:
      x = Activation(activation)(x)
    return x

def getModel():
    inputShape = (32,128,1)
    rnnUnits = 256
    maxStringLen = 32
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    inner = res_block(inputs,64)
    inner = res_block(inner,64)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName1')(inner)
    inner = res_block(inner,128)
    inner = res_block(inner,128)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName2')(inner)
    inner = res_block(inner,256)
    inner = res_block(inner,256)
    inner = MaxPooling2D(pool_size = (1,2),strides = (2,2), name = 'MaxPoolName4')(inner)
    inner = res_block(inner,512)
    inner = res_block(inner,512)
    inner = MaxPooling2D(pool_size = (1,2), strides = (2,2), name = 'MaxPoolName6')(inner)
    inner = res_block(inner,512)

    inner = Reshape(target_shape = (maxStringLen,rnnUnits), name = 'reshape')(inner)

    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM1F')(inner)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM1B')(inner)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)

    LS1 = average([LSF,LSB])
    LS1 = BatchNormalization()(LS1)

    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM2F')(LS1)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM2B')(LS1)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)

    LS2 = concatenate([LSF,LSB])
    LS2 = BatchNormalization()(LS2)
    yPred = Dense(len(unicodes)+1,kernel_initializer='he_normal',name='dense2')(LS2)
    yPred = Activation('softmax')(yPred)
    return Model(inputs=[inputs], outputs=yPred)

model = getModel()
model.load_weights('ResNetBestNew_weights.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('ResNetBestNew.tflite', 'wb') as f:
  f.write(tflite_model)
#%%