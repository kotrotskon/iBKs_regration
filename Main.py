from __future__ import absolute_import, division, print_function, unicode_literals

import os
import csv
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_docs.plots
import pandas as pd

print(tf.version.VERSION)

column_names = ['id', 'point', 'rssi_a', 'rssi_b', 'rssi_c', 'rssi_d', 'rssi_e',
                'distance_a', 'distance_b', 'distance_c', 'distance_d', 'distance_e']

dataset_path = 'measurements.csv'
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=";", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)
test_dataset = dataset


test_labels_a = test_dataset.pop('distance_a')
test_labels_b = test_dataset.pop('distance_b')
test_labels_c = test_dataset.pop('distance_c')
test_labels_d = test_dataset.pop('distance_d')
test_labels_e = test_dataset.pop('distance_e')
points = test_dataset.pop('point')

test_dataset.drop(["id"], axis=1, inplace=True)

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


def get_position_from_distance(distance):
    switcher = {
        1: [50, 50],
        2: [100, 50],
        3: [150, 50],
        4: [200, 50],
        5: [250, 50],
        6: [300, 50],
        7: [350, 50],
        8: [400, 50],
        9: [450, 50],
        10: [500, 50],
        11: [550, 50],
        12: [50, 100],
        13: [100, 100],
        14: [150, 100],
        15: [200, 100],
        16: [250, 100],
        17: [300, 100],
        18: [350, 100],
        19: [400, 100],
        20: [450, 100],
        21: [500, 100],
        22: [550, 100],
        23: [50, 150],
        24: [100, 150],
        25: [150, 150],
        26: [200, 150],
        27: [250, 150],
        28: [300, 150],
        29: [350, 150],
        30: [400, 150],
        31: [450, 150],
        32: [500, 150],
        33: [550, 150],
        34: [50, 200],
        35: [100, 200],
        36: [150, 200],
        37: [200, 200],
        38: [250, 200],
        39: [300, 200],
        40: [350, 200],
        41: [400, 200],
        42: [450, 200],
        43: [500, 200],
        44: [550, 200],
        45: [50, 250],
        46: [100, 250],
        47: [150, 250],
        48: [200, 250],
        49: [250, 250],
        50: [300, 250],
        51: [350, 250],
        52: [400, 250],
        53: [450, 250],
        54: [500, 250],
        55: [550, 250],
        56: [50, 300],
        57: [100, 300],
        58: [150, 300],
        59: [200, 300],
        60: [250, 300],
        61: [300, 300],
        62: [350, 300],
        63: [400, 300],
        64: [450, 300],
        65: [500, 300],
        66: [550, 300],
        67: [50, 350],
        68: [100, 350],
        69: [150, 350],
        70: [200, 350],
        71: [250, 350],
        72: [300, 350],
        73: [350, 350],
        74: [400, 350],
        75: [450, 350],
        76: [500, 350],
        77: [550, 350],
        78: [50, 400],
        79: [100, 400],
        80: [150, 400],
        81: [200, 400],
        82: [250, 400],
        83: [300, 400],
        84: [350, 400],
        85: [400, 400],
        86: [450, 400],
        87: [500, 400],
        88: [550, 400],
        89: [50, 450],
        90: [100, 450],
        91: [150, 450],
        92: [200, 450],
        93: [250, 450],
        94: [300, 450],
        95: [350, 450],
        96: [400, 450],
        97: [450, 450],
        98: [500, 450],
        99: [550, 450],
        100: [50, 500],
        101: [100, 500],
        102: [150, 500],
        103: [200, 500],
        104: [250, 500],
        105: [300, 500],
        106: [350, 500],
        107: [400, 500],
        108: [450, 500],
        109: [500, 500],
        110: [550, 500]
    }

    return switcher.get(distance)


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

print("Model A")
print(model_a)

print("Normed test data: ")
print(normed_test_data)

print("Test labels A")
test_points = points.tolist()
print(test_labels_a)
print(test_labels_b)
print(test_labels_c)
print(test_labels_d)
print(test_labels_e)
real_positions = []

for i in range(len(test_points)):
    real_positions.append(get_position_from_distance(test_points[i]))

test_predictions_a = model_a.predict(normed_test_data).flatten()
test_predictions_b = model_b.predict(normed_test_data).flatten()
test_predictions_c = model_c.predict(normed_test_data).flatten()
test_predictions_d = model_d.predict(normed_test_data).flatten()
test_predictions_e = model_e.predict(normed_test_data).flatten()

print("Test Predictions A")
print(test_predictions_a)
print(test_predictions_b)
print(test_predictions_c)
print(test_predictions_d)
print(test_predictions_e)

with open('predictions_file.csv', mode='w') as predictions_file:
    for i in range(len(test_predictions_a)):
        predictions_writer = csv.writer(predictions_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predictions_writer.writerow([test_predictions_a[i], test_predictions_b[i], test_predictions_c[i],
                                     test_predictions_d[i], test_predictions_e[i], real_positions[i]
                                     ])

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
