"""Example of Estimator for Iris plant dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from numpy import array,loadtxt
#from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
from time import time
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


startTime = time()
filename = 'data\digitsX.dat'
X = loadtxt(filename,delimiter=',')
keyname = 'data\digitsY.dat'
Y = loadtxt(keyname,delimiter=',')
x_train, x_test, y_train, y_test = model_selection.train_test_split(
  X,Y, test_size=0.2, random_state=42)
del X, Y
def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_L1], activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(hparams[HP_L2], activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy

'''def train_test_model_relu(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_L1], activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(hparams[HP_L2], activation=tf.nn.relu),
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(x_train, y_train, epochs=5) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy

def train_test_model_tanh(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_L1], activation=tf.nn.tanh),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(hparams[HP_L2], activation=tf.nn.tanh),
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(x_train, y_train, epochs=5) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy

def train_test_model_softmax(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_L1], activation=tf.nn.softmax),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(hparams[HP_L2], activation=tf.nn.softmax),
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(x_train, y_train, epochs=5) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy'''

if __name__ == '__main__':
  p1 = array(range(12))*5+75
  p1 = p1.tolist()
  p2 = array(range(12))*5+75
  p2 = p2.tolist()
  HP_L1 = hp.HParam('num_units', hp.Discrete(p1))
  HP_L2 = hp.HParam('num_units', hp.Discrete(p2))
  #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', ]))

  METRIC_ACCURACY = 'accuracy'

  
  session_num = 0
  record = []
  accrec = []
  for n1 in HP_L1.domain.values:
    for n2 in HP_L2.domain.values:
      
      hparams = {
          HP_L1: n1,
          HP_L2: n2,
      }
      run_name = "run-%d" % session_num
      acc1 = train_test_model(hparams)
      
      #print('--- Starting trial: %s' % run_name)
      #print({h.name: hparams[h] for h in hparams})
      #print('accuracy = ', train_test_model(hparams))
      record.append([n1,n2,acc1])
      accrec.append(acc1)
      session_num += 1
record = array(record)

trueacc = accrec.index(max(accrec))
print (record[trueacc,:])
### Plot time!



plt.subplot(2, 2, 1)

#plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.plot(record[:, 0], record[:, 2], 'b-')
plt.title("Dependant on Layer 1")
plt.axis('tight')
#ax.scatter3D(xdata, ydata, zdata, c=zdata,)

plt.subplot(2, 2, 2)


plt.plot(record[:, 1], record[:, 2], 'b-')
plt.title('Dependant on Layer 2')
plt.axis('tight')
plt.subplot(2, 2, 3)
'''
#plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.plot(record[:, 0], record[:, 4], 'b-')
plt.title("1_layer tanh")
plt.axis('tight')
#ax.scatter3D(xdata, ydata, zdata, c=zdata,)

plt.subplot(2, 2, 4)


plt.plot(record[:, 0], record[:, 3], 'b-')
plt.title('1-layer Softmax')
plt.axis('tight')
'''
plt.show()

    