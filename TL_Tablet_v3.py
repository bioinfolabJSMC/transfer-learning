# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 20:25:53 2021

@author: roger
"""


# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, LeakyReLU
from keras.utils import plot_model
from keras import backend as K
from keras import regularizers
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat
import keras
import numpy as np

data =loadmat('D:/工作/TransferLearning/nir_shootout_MT.mat')

Xm, Xs = data['Xm_502'], data['Xs_502']
Xmtest,  Xstest  = data['Xtestm'], data['Xtests']
ytrain, ytest = data['ycal_502'], data['ytest']

ytrain=ytrain
ytest=ytest

wv = data['wv']
# input image dimensions
xlength = len(Xm[0]) 
ylength = 1
img_rows, img_cols = 1, xlength


#%%
batch_size = 8
num_classes = ylength
epochs = 3000

# input spectra of slave instrument 
out_train = ytrain 
in_train  = Xs.reshape(Xs.shape[0], img_cols, img_rows)

in_test  = Xstest.reshape(Xstest.shape[0], img_cols, img_rows)
input_shape = (img_cols,img_rows)


# load model for the master instrument 
modelname='model20210812151304.h5'
with open(modelname[0:20] + 'json', 'r') as json_file:
    model_json = json_file.read()
base_model = model_from_json(model_json)
base_model.load_weights(modelname)
base_model.trainable = False

base_model.summary()

resnet_model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_20').output)
resnet_model.trainable = False


# build new model
new_model=Sequential()
new_model.add(resnet_model)
new_model.add(Dense(16))
new_model.add(Dense(8))
new_model.add(Dense(num_classes))

new_model.summary()

optimizer = keras.optimizers.Adam(lr=0.0001)   
new_model.compile(loss='mse', optimizer=optimizer)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

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

history=new_model.fit(in_train, out_train,
          batch_size = batch_size,
          epochs     = epochs,
          verbose=2,
          validation_split=0.1,
          callbacks = [savebestmodel]
          ) 

# 保存模型
model_json = new_model.to_json()
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
pls1 = PLSRegression(n_components=6,scale=True)
pls1.fit(Xm, ytrain)
yhat_train_pls1 = pls1.predict(Xm)
yhat_test_pls1 = pls1.predict(Xmtest)

pls2 = PLSRegression(n_components=6,scale=True)
pls2.fit(Xs, ytrain)
yhat_train_pls2 = pls2.predict(Xs)
yhat_test_pls2 = pls2.predict(Xstest)


from sklearn.metrics import mean_squared_error
from math import sqrt
rmsep1 = sqrt(mean_squared_error(ytest, yhat_test))
rp21=(1-mean_squared_error(ytest,yhat_test)/np.var(ytest))
rmsec1 = sqrt(mean_squared_error(ytrain, yhat_train))
rc21=(1-mean_squared_error(ytrain,yhat_train)/np.var(ytrain))

rmsep1_pls1 = sqrt(mean_squared_error(ytest, yhat_test_pls1))
rp21_pls1=(1-mean_squared_error(ytest,yhat_test_pls1)/np.var(ytest))
rmsec1_pls1 = sqrt(mean_squared_error(ytrain, yhat_train_pls1))
rc21_pls1=(1-mean_squared_error(ytrain,yhat_train_pls1)/np.var(ytrain))

rmsep1_pls2 = sqrt(mean_squared_error(ytest, yhat_test_pls2))
rp21_pls2=(1-mean_squared_error(ytest,yhat_test_pls2)/np.var(ytest))
rmsec1_pls2 = sqrt(mean_squared_error(ytrain, yhat_train_pls2))
rc21_pls2=(1-mean_squared_error(ytrain,yhat_train_pls2)/np.var(ytrain))

print('--------------------------------------------------------------')
print('-------CNN-------')
print('RMSEC1:',str('%.3f' %(rmsec1)),'  Rc21:',str('%.3f' %(rc21)),'  RMSEP1:' ,str('%.3f' %(rmsep1)),
      '  Rp21:',str('%.3f' %(rp21)))
print('----------------------------PLS-------------------------------')
print('------MASTE------')
print('RMSEC1:',str('%.3f' %(rmsec1_pls1)),'  Rc21:',str('%.3f' %(rc21_pls1)),'  RMSEP1:' ,str('%.3f' %(rmsep1_pls1)),
      '  Rp21:',str('%.3f' %(rp21_pls1)))
print('------SLAVE------')
print('RMSEC2:',str('%.3f' %(rmsec1_pls2)),'  Rc21:',str('%.3f' %(rc21_pls2)),'  RMSEP1:' ,str('%.3f' %(rmsep1_pls2)),
      '  Rp21:',str('%.3f' %(rp21_pls2)))
print('--------------------------------------------------------------')



