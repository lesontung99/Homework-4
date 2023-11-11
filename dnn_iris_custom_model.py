#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 
#  Modified and annotated by Eric Eaton in 2017 for CIS 419/519 at Penn
#
"""Example of Estimator for Iris plant dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf


X_FEATURE = 'x'  # Name of the input feature.


def my_model(features, labels, mode):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  
  # Note how the variable "net" is repeatedly used as the input to the next layer, 
  # then updated.  This creates a net that looks like:
  #   features -> layer10 -> dropout -> layer20 -> dropout -> layer10 -> dropout -> argmax_logits
  # where "layer##" is a fully connected relu layer with ## units
  
  # Create three fully connected layers respectively of size 10, 20, and 10 with
  # each layer having a dropout probability of 0.1
  net = features[X_FEATURE]
  for units in [10, 20, 10]:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.1)

  # Compute logits (1 per class)
  logits = tf.layers.dense(net, 3, activation=None)

  # Compute predictions via the argmax over the logits
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Convert the labels to a one-hot tensor of shape (length of features, 3) and
  # with a on-value of 1 for each one-hot vector of length 3.
  onehot_labels = tf.one_hot(labels, 3, 1, 0)
  
  # Compute the loss as the cross-entropy of the softmax outputs
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Create training optimizer, using adagrad to minimize the cross entropy loss
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  
  return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

  # load the iris dataset, and split into training/testing sets
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  # initialize the classifier to use our custom model
  classifier = tf.estimator.Estimator(model_fn=my_model)

  # train the model for 1000 steps
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=1000)

  # predict classes for the test data
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # compute the accuracy via sklearn's built-in function
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # compute the accuracy via TensorFlow's built-in function
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()