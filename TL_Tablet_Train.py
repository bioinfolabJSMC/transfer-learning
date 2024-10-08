# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:37:01 2021

@author: roger
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, LeakyReLU

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat
import keras
import numpy as np

data =loadmat('E:/工作/A TransferLearning/nir_shootout_MT.mat')

Xm, Xs = data['Xm_502'], data['Xs_502']
Xmtest,  Xstest  = data['Xtestm'], data['Xtests']
ytrain, ytest = data['ycal_502'], data['ytest']


wv = data['wv']
# input image dimensions
xlength = len(Xm[0]) 
ylength = 1
img_rows, img_cols = 1, xlength


#%%
batch_size = 8
num_classes = ylength
epochs = 10000

out_train = ytrain 
in_train  = Xm.reshape(Xm.shape[0], img_cols, img_rows)

in_test  = Xmtest.reshape(Xmtest.shape[0], img_cols, img_rows)
input_shape = (img_cols,img_rows)

conv1d = keras.layers.convolutional.Conv1D(filters=8, 
                                           kernel_size=10, 
                                           padding='same',
#                                           activation= None, 
                                           input_shape=input_shape
                                           );
conv1d2 = keras.layers.convolutional.Conv1D(filters=16, 
                                            kernel_size=20, 
                                            padding='same',
#                                            activation= None
                                            );
conv1d3 = keras.layers.convolutional.Conv1D(filters=32, 
                                            kernel_size=30, 
                                            padding='same',
#                                            activation= None
                                            );
max_pooling_layer = keras.layers.MaxPool1D(pool_size=2);

model = Sequential()
model.add(conv1d)
model.add(Activation('relu'))
#model.add(max_pooling_layer)
model.add(conv1d2)
model.add(Activation('relu'))
#model.add(max_pooling_layer)
model.add(conv1d3)
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(num_classes))


optimizer = keras.optimizers.Adam(lr=0.0001)   
model.compile(loss='mse', optimizer=optimizer)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

import time
timestr = time.strftime("%Y%m%d%H%M%S", time.localtime())
modelname = 'model'+timestr+'.h5'
earlystop = EarlyStopping(monitor = 'loss_val', 
                         patience = 300,
                         restore_best_weights = True)
savebestmodel = ModelCheckpoint(modelname, 
                                monitor = 'loss', 
                                verbose = 1, 
                                save_best_only = True, 
                                mode = 'auto')

history=model.fit(in_train, out_train,
          batch_size = batch_size,
          epochs     = epochs,
          verbose=2,
          validation_split=0.1,
          callbacks = [savebestmodel]
          ) 

# 保存模型
model_json = model.to_json()
with open(modelname[0:20] + 'json', 'w') as json_file:
    json_file.write(model_json)

# 读取模型
with open(modelname[0:20] + 'json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(modelname)

#score = model.evaluate(x_test, y_test, verbose=0)
yhat_test = model.predict(in_test, batch_size=1)
yhat_train = model.predict(in_train, batch_size=1)

model.summary()

#%% figure
plt.figure()
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#
plt.figure()
plt.plot(ytrain, yhat_train, 'bo')
plt.plot(ytest, yhat_test,'ro')
plt.plot(ytrain, ytrain,'k-')
plt.ylabel('Predicted')
plt.xlabel('Reference')
plt.legend(['training set', 'test set'], loc='upper left')
plt.show()

#%% PLS
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=6,scale=True)
pls.fit(Xm, ytrain)
yhat_train_pls = pls.predict(Xm)
yhat_test_pls = pls.predict(Xmtest)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmsep1 = sqrt(mean_squared_error(ytest, yhat_test))
rp21=(1-mean_squared_error(ytest,yhat_test)/np.var(ytest))
rmsec1 = sqrt(mean_squared_error(ytrain, yhat_train))
rc21=(1-mean_squared_error(ytrain,yhat_train)/np.var(ytrain))

rmsep1_pls = sqrt(mean_squared_error(ytest, yhat_test_pls))
rp21_pls=(1-mean_squared_error(ytest,yhat_test_pls)/np.var(ytest))
rmsec1_pls = sqrt(mean_squared_error(ytrain, yhat_train_pls))
rc21_pls=(1-mean_squared_error(ytrain,yhat_train_pls)/np.var(ytrain))


print('--------------------------------------------------------')
print('RMSEC1:',str('%.3f' %(rmsec1)),'  Rc21:',str('%.3f' %(rc21)),'  RMSEP1:' ,str('%.3f' %(rmsep1)),
      '  Rp21:',str('%.3f' %(rp21)))
print('RMSEC1:',str('%.3f' %(rmsec1_pls)),'  Rc21:',str('%.3f' %(rc21_pls)),'  RMSEP1:' ,str('%.3f' %(rmsep1_pls)),
      '  Rp21:',str('%.3f' %(rp21_pls)))
print('--------------------------------------------------------')

