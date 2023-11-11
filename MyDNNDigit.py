
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from numpy import array,loadtxt
#from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import tensorflow as tf
from time import time
from tensorboard.plugins.hparams import api as hp


startTime = time()
filename = 'Questions\CIS419-master\Assignment4\hw4_skeleton_20171106\data\digitsX.dat'
X = loadtxt(filename,delimiter=',')
keyname = 'Questions\CIS419-master\Assignment4\hw4_skeleton_20171106\data\digitsY.dat'
Y = loadtxt(keyname,delimiter=',')
x_train, x_test, y_train, y_test = model_selection.train_test_split(
  X,Y, test_size=0.2, random_state=42)
del X, Y
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(95, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
model.fit(x_train, y_train,epochs=25)
acc = model.evaluate(x_test,y_test)
print("Accuracy:" ,acc)