import numpy as np
import tensorflow as tf

from tensorflow import keras

# A simple (my first) implementation of an AI for the fashion_mnist dataset; achieves 0.3493 loss

dataset = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images = training_images / 255
test_images = test_images / 255

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                         keras.layers.Dense(128, activation=tf.nn.relu),
                         keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
