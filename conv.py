import numpy as np
import tensorflow as tf

from tensorflow import keras

# A simple implementation of an AI for the fashion_mnist dataset. Instead of using only a dense layer, it uses
# convolution layers; achieves 0.2569 loss

dataset = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images = training_images / 255
test_images = test_images / 255

# there was a bug in my code where it would say expected ndim=3, found ndim=4, this seems to fix it
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
