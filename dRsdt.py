# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:08:44 2024

@author: mx54
"""

import pandas  as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import layers
from keras import models
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras import backend as K


plt.style.use('default')

############################
#import the dataset, check the dataset first 5 dataset values, split the data into 'X' inputs and 'y' outputs
############################

import os;
###################################
# enter where you store your dataset, use double backslash as given in example
##################################################
path="z:\\working\\2.1"
os.chdir(path)
os.getcwd()
seed=10
#Variables
data = pd.read_csv('dRsdt.csv')
data.head(5)
X = data.iloc[:5000, 0:3]
y = data.iloc[:5000, 3:4]
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

############################
#Scalerize the data and split the dataset into train test and validation.
#important- check document for links
############################
scaler_x = MaxAbsScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X))
xscale=scaler_x.transform(X)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)
data.to_csv(index=True)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
print("Number transactions X_val dataset: ", X_val.shape)
print("Number transactions y_val dataset: ", y_val.shape)

############################
#set seed values important to reproduce the results
# useful to reproduce results- will be discussed later
############################

# Seed value
# Apparently you may use different seed values at each stage
seed_value=10

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

############################
#The ANN model
############################

model=Sequential()
model.add(Dense(64, input_dim=3, kernel_initializer='normal', activation=tf.keras.layers.LeakyReLU(alpha=0.01), name= 'input'))
model.add(Dense(32, activation='sigmoid', name= 'first_layer'))
model.add(Dense(16, activation='sigmoid', name= 'second_layer'))
model.add(Dense(1, activation='linear', name= 'output'))#linear
model.summary()

############################
#compile the model with loss and optimizer function.
############################

def custom_loss(y_true, y_pred):
    max_value = tf.reduce_max(y_true)
    loss = tf.abs(y_pred - y_true) / max_value
    return loss

sgd = optimizers.SGD(learning_rate=0.05, decay=2e-8, momentum=0.9, nesterov=True, clipvalue=0.5)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

############################
#early stopping is used to reduce the computer power needed to complete the model once learned.-hyperparameters will be discussed later on
############################

num_epoch = 50

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=np.round(num_epoch*0.3,1))
checkpointer = ModelCheckpoint(filepath='Model_weights.h5', verbose=1, save_best_only=True)

############################
#fit the ANN model to the validation dataset and check.
############################

history = model.fit(X_train, y_train, epochs=num_epoch, batch_size=32, verbose=1, validation_data=(X_val, y_val), shuffle=True, callbacks=[es, checkpointer])

############################
#history keys are needed to check what names your system has for loss and Val_loss-important to plot
############################

print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

############################
#plot the loss function, import matplotlib for this
############################

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#################
#change 'loss' and 'val_loss' depending upon what it is for your computer as printed above
#################
plt.plot(epochs, loss, 'k', label='Training loss')
plt.plot(epochs, val_loss, 'k--', marker='D', markeredgecolor='black', markevery=50, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('MAE Loss')
plt.legend()
plt.show()

############################
#evaluate your model score
############################

score = model.evaluate(X_test, y_test)
print("Test Loss: %.2f%%" % (score[0]*100))
print("Test Accuracy: %.2f%%" % (score[1]*100))

############################
#introduce a new dataset value to check your model on totally "unseen" value-you can generate as many new datapoints and check the prediction by changing the first six values.
############################

Xnew = np.array([17300,	0,	5.65E-01]).reshape(1, -1)
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew, ynew),'real y =	5.71')
Xnew1 = np.array([17154.54373,	0.122836467,	5.73E-01]).reshape(1, -1)
Xnew1= scaler_x.transform(Xnew1)
ynew1= model.predict(Xnew1)
ynew1 = scaler_y.inverse_transform(ynew1)
Xnew1 = scaler_x.inverse_transform(Xnew1)
print("X=%s, Predicted=%s" % (Xnew1, ynew1),'real y =	59.1')

weights = model.layers[0].get_weights()[1]
np.savetxt('dRsdt_bias-input_layer.txt', weights, fmt='%1.4e', delimiter=',')
weights1 = model.layers[0].get_weights()[0]
np.savetxt('dRsdt_weights-input_layer.txt', weights1, fmt='%1.4e', delimiter=',')

weights2 = model.layers[1].get_weights()[1]
np.savetxt('dRsdt_bias-first_layer.txt', weights2, fmt='%1.4e', delimiter=',')
weights3 = model.layers[1].get_weights()[0]
np.savetxt('dRsdt_weights-first_layer.txt', weights3, fmt='%1.4e', delimiter=',')

weights4 = model.layers[2].get_weights()[1]
np.savetxt('dRsdt_bias-second_layer.txt', weights4, fmt='%1.4e', delimiter=',')
weights5 = model.layers[2].get_weights()[0]
np.savetxt('dRsdt_weights-second_layer.txt', weights5, fmt='%1.4e', delimiter=',')

weights6 = model.layers[3].get_weights()[1]
np.savetxt('dRsdt_bias-third_layer.txt', weights6, fmt='%1.4e', delimiter=',')
weights7 = model.layers[3].get_weights()[0]
np.savetxt('dRsdt_weights-third_layer.txt', weights7, fmt='%1.4e', delimiter=',')


'''
weights2 = model.layers[3].get_weights()[1]
np.savetxt('dinputOutput_bias-output_layer.txt', weights2 ,fmt='%1.4e', delimiter=',')
weights3 = model.layers[3].get_weights()[0]
np.savetxt('dinputOutput_weights-output_layer.txt', weights3 ,fmt='%1.4e', delimiter=',')
'''
############################
#generate a scatter plot agaisnt actualvspredicted to show how good your model is
############################

def scatter_y(y_true, y_pred):
    """Scatter-plot the predicted vs true number
   
    Plots:
       * predicted vs true number
       * perfect agreement line
       * +2/-2 number dotted lines

    Returns the root mean square of the error
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(y_true, y_pred, '.k')
    ax.scatter(y_true, y_pred, alpha=0.5)
   
    ax.plot([0, 1], [0, 1], '--r', linewidth=2)
    ax.plot([0, 1], [0.05, 1.05], ':r')
    ax.plot([0.05, 1.05], [0, 1], ':r')
   
    rms = (y_true-y_pred).std()
   
    ax.text(0.8, 0.1,
            "RMSE = %.2g" % rms,
            ha='right', va='bottom')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
   
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
   
    return rms
y1_pred = model.predict(X_val)
scatter_y(y_val, y1_pred)
y2_pred = model.predict(X_test)
scatter_y(y_test,y2_pred)
plt.show()
############################
#print all the mterics to check your model
############################

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y2_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y2_pred))
print('Root Mean Squared Error test:', np.sqrt(metrics.mean_squared_error(y_test, y2_pred)))