from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import pathlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

column_names = ['id', 'point', 'rssi_a', 'rssi_b', 'rssi_c', 'rssi_d', 'rssi_e',
                'distance_a', 'distance_b', 'distance_c', 'distance_d', 'distance_e', 'timestamp']

dataset_path = 'measurements.csv'
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=";", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels_a = train_dataset.pop('distance_a')
test_labels_a = test_dataset.pop('distance_a')
train_labels_b = train_dataset.pop('distance_b')
test_labels_b = test_dataset.pop('distance_b')
train_labels_c = train_dataset.pop('distance_c')
test_labels_c = test_dataset.pop('distance_c')
train_labels_d = train_dataset.pop('distance_d')
test_labels_d = test_dataset.pop('distance_d')
train_labels_e = train_dataset.pop('distance_e')
test_labels_e = test_dataset.pop('distance_e')

train_labels = train_labels_e
test_labels = test_labels_e

train_dataset.drop(["id", "point", "timestamp"], axis=1, inplace=True)
test_dataset.drop(["id", "point", "timestamp"], axis=1, inplace=True)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)

train_labels_stats = train_labels.describe()
train_labels_stats = train_labels_stats.transpose()
print(train_labels)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_train_data)
print(normed_test_data)
print(test_labels)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


tf.keras.backend.clear_session()
tf.random.set_seed(60)

model = build_model()

model.summary()

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


EPOCHS = 1000

history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split=0.2,
                    batch_size=1024,
                    validation_data=(normed_test_data, test_labels),
                    verbose=1,
                    callbacks=[cp_callback])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 100])
plt.ylabel('MAE [Distance]')
plt.show()

plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 5000])
plt.ylabel('MSE [Distance^2]')
plt.show()


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split=0.2, verbose=0,
                    callbacks=[cp_callback, early_stop, tfdocs.modeling.EpochDots()])

model.save('saved_model/my_model')

plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 150])
plt.ylabel('MAE [Distance]')
plt.show()


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} Distance".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [Distance]")
_ = plt.ylabel("Count")
plt.show()


print(normed_test_data)
print(test_predictions)
print(test_labels)









