import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.initializers import TruncatedNormal, Constant

def step_decay(epochs):
    x = 1e-4
    if 500 < epochs: 
        x = 1e-5
        print("----{}----".format(x))
    return x

def conv2d(filters, kernel_size, strides=(1, 1), padding='same', bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters, kernel_size, strides=strides, padding=padding,
        activation='relu', kernel_initializer=trunc, bias_initializer=cnst, **kwargs
    )   

def dense(units, activation='relu'):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units, activation=activation,
        kernel_initializer=trunc, bias_initializer=cnst,
    )   

def AlexNet(input_shape, num_classes):
    model = Sequential()

    #conv1
    model.add(conv2d(96, 11, strides=(4, 4), padding='valid', bias_init=0,
        input_shape=input_shape))   #一層目からsameだと容量がでかくなるから
    #pool1
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #conv2
    model.add(conv2d(256, 5)) 
    #pool2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #conv3
    model.add(conv2d(384, 3, bias_init=0))
    #conv4
    model.add(conv2d(384, 3)) 
    #conv5
    model.add(conv2d(256, 3)) 
    #pool5
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #fc6
    model.add(Flatten())    #一次元にする
    model.add(dense(4096))
    model.add(Dropout(0.5))
    #fc7
    model.add(dense(4096))
    model.add(Dropout(0.5))

    #fc8
    model.add(dense(num_classes, activation='linear'))

    return model

class Trainer():
    
    def __init__(self, log_dir, model, loss, verbose):
        self._target = model
        self.lr = LearningRateScheduler(step_decay)
        self.optimizer = Adam(lr=self.lr)
        self._target.compile(loss=loss, optimizer=self.optimizer, metrics=['mae'])
        self.verbose = verbose
        self.log_dir = log_dir

    def train(self, x_train, y_train, batch_size, epochs, validation_data):
        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=validation_data,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.log_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'), monitor='val_loss', period=50)
            ],
            verbose=self.verbose
        )

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
#keras.backend.set_session(tf.Session(config=config))

log_dir = './../../Data/Models/'
if os.path.exists(log_dir):
    import shutil
    shutil.rmtree(log_dir)
os.mkdir(log_dir)

image_shape = (256, 128, 3)     #
num_classes = 1     #出力する数（材質だったら3になる）

model = AlexNet(image_shape, num_classes)
model.summary() #modelの要約

x_train = np.load('./../../Data/Dataset_npy/x_train.npy')
print(x_train)
x_test = np.load('./../../Data/Dataset_npy/x_test.npy')
print(x_test)
y_train = np.load('./../../Data/Dataset_npy/y_train.npy')
print(y_train)
y_test = np.load('./../../Data/Dataset_npy/y_test.npy')
print(y_test)

trainer = Trainer(log_dir=log_dir,model=model, loss='mean_squared_error', verbose=1)    #rmsprop
trainer.train(x_train, y_train, batch_size=128, epochs=3, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
