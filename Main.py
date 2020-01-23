from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_docs.plots
import pandas as pd

print(tf.version.VERSION)

column_names = ['id', 'point', 'rssi_a', 'rssi_b', 'rssi_c', 'rssi_d', 'rssi_e',
                'distance_a', 'distance_b', 'distance_c', 'distance_d', 'distance_e', 'timestamp']

dataset_path = './measurements.csv'
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=";", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

test_labels_a = test_dataset.pop('distance_a')
test_labels_b = test_dataset.pop('distance_b')
test_labels_c = test_dataset.pop('distance_c')
test_labels_d = test_dataset.pop('distance_d')
test_labels_e = test_dataset.pop('distance_e')

test_dataset.drop(["id", "point", "timestamp"], axis=1, inplace=True)

train_stats = test_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(test_dataset)

checkpoint_path_a = 'training/training_a/cp_a.ckpt'
checkpoint_dir_a = os.path.dirname(checkpoint_path_a)

checkpoint_path_b = 'training/training_b/cp_b.ckpt'
checkpoint_dir_b = os.path.dirname(checkpoint_path_b)

checkpoint_path_c = 'training/training_c/cp_c.ckpt'
checkpoint_dir_c = os.path.dirname(checkpoint_path_c)

checkpoint_path_d = 'training/training_d/cp_d.ckpt'
checkpoint_dir_d = os.path.dirname(checkpoint_path_d)

checkpoint_path_e = 'training/training_e/cp_e.ckpt'
checkpoint_dir_e = os.path.dirname(checkpoint_path_e)


def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[5]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


latest_a = tf.train.latest_checkpoint(checkpoint_dir_a)
print(latest_a)

latest_b = tf.train.latest_checkpoint(checkpoint_dir_b)
print(latest_b)

latest_c = tf.train.latest_checkpoint(checkpoint_dir_c)
print(latest_c)

latest_d = tf.train.latest_checkpoint(checkpoint_dir_d)
print(latest_d)

latest_e = tf.train.latest_checkpoint(checkpoint_dir_e)
print(latest_e)

# Create a new model instance
model_a = create_model()
model_b = create_model()
model_c = create_model()
model_d = create_model()
model_e = create_model()

# Load the previously saved weights
model_a.load_weights(latest_a)
model_b.load_weights(latest_b)
model_c.load_weights(latest_c)
model_d.load_weights(latest_d)
model_e.load_weights(latest_e)

print(model_a)

print(normed_test_data)
print(test_labels_a)
print(test_labels_b)
print(test_labels_c)
print(test_labels_d)
print(test_labels_e)


test_predictions_a = model_a.predict(normed_test_data).flatten()
test_predictions_b = model_b.predict(normed_test_data).flatten()
test_predictions_c = model_c.predict(normed_test_data).flatten()
test_predictions_d = model_d.predict(normed_test_data).flatten()
test_predictions_e = model_e.predict(normed_test_data).flatten()

print(test_predictions_a)
print(test_predictions_b)
print(test_predictions_c)
print(test_predictions_d)
print(test_predictions_e)

a = plt.axes(aspect='equal')
plt.scatter(test_labels_a, test_predictions_a)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(test_labels_b, test_predictions_b)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(test_labels_c, test_predictions_c)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(test_labels_d, test_predictions_d)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(test_labels_e, test_predictions_e)
plt.xlabel('True Values [Distance]')
plt.ylabel('Predictions [Distance]')
lims = [0, 700]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


