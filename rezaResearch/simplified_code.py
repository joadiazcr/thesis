"""
@author: Jorge

Simplified version of Reza's code
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

start_time = time.time()

## Dataset nomalization
def norm(dataset, mean, std):
  return (dataset - mean) / std


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# Load data
test_dataset= pd.read_csv('test_dataset.csv')
train_dataset= pd.read_csv('train_dataset.csv')
dataset= pd.read_csv('dataset.csv')

#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset.pop('Error')
train_dataset_Time=train_dataset.pop('Time')

test_dataset.pop('P')
test_dataset.pop('D')
test_dataset.pop('Error')
test_dataset_Time=test_dataset.pop('Time')

train_labels = train_dataset.pop('Energy')
test_labels = test_dataset.pop('Energy')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
mean = train_stats['mean']
std = train_stats['std']
normed_train_data = norm(train_dataset, mean, std)
normed_test_data = norm(test_dataset, mean, std)

# Build model
print ("Building the NN model...")
input_nodes = len(train_dataset.keys())

#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
#
#  model = keras.Sequential([
#    layers.Dense(10, activation=tf.nn.relu, input_shape=[input_nodes]),
#    layers.Dense(1)
#  ])
#
#  optimizer = tf.keras.optimizers.RMSprop(0.001)
#  model.compile(loss='mean_absolute_percentage_error',
#              optimizer=optimizer,
#              metrics=['mean_squared_error','mean_absolute_percentage_error'])

model = keras.Sequential([
  layers.Dense(10, activation=tf.nn.relu, input_shape=[input_nodes]),
  layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_absolute_percentage_error',
            optimizer=optimizer,
            metrics=['mean_squared_error','mean_absolute_percentage_error'])
print ("Model built")

## Train the NN
EPOCHS = 15
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1400)
#history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
#                    validation_split = 0.2, verbose=1, callbacks=[early_stop, PrintDot()
#])
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS)


## Test the NN
loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} Energy".format(mse))
print("MSE = %s" %mse)
print("MAPE = %s" %mape)
