import os

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import tensorflow as tf
import keras
from keras import models


LARGE_DATA_DIR = 'notMNIST_large'
SMALL_DATA_DIR = 'notMNIST_small'
DATA_DIR = ""

def partial_linear(x):
    return tf.maximum(x, 0.1 * x)

# Normalize the images
def normalize(data):
    return (data.astype(np.float32) - 128.0) / 128.0


# Remove duplicates from the dataset
def remove_duplicates(data):
    _, index = np.unique(data[:, 1:], axis=0, return_index=True)
    return data[index]


def load_images_with_progress(image_dir, class_label):
    image_data = []
    image_labels = []
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    for filename in tqdm(files, desc=f"Загрузка {class_label}", unit="img"):
        path = os.path.join(image_dir, filename)
        try:
            with Image.open(path) as img:
                image_data.append(np.array(img))
                image_labels.append(ord(class_label) - ord('A'))
        except UnidentifiedImageError:
            continue
        except Exception as error:
            print(type(error), error)

    return image_data, image_labels


def load_data():
    data = []
    labels = []
    for label, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
        folder = os.path.join(os.path.expanduser("~"), DATA_DIR, letter)
        try:
            current_data, current_labels = load_images_with_progress(folder, letter)
            data.extend(current_data)
            labels.extend(current_labels)
        except Exception as error:
            print(error)
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape((-1, 28, 28, 1))
    data = remove_duplicates(np.hstack([labels.reshape(-1, 1), data.reshape(-1, 28 * 28)]))

    labels = data[:, 0].astype(np.int32)
    data = data[:, 1:].reshape((-1, 28, 28, 1))

    return data, labels


def conv_neural_network(x_train, y_train):
    # Create the convolutional neural network
    model = models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        # First Convolutional Layer
        keras.layers.Conv2D(32, (3, 3), activation=partial_linear),
        # keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Layer
        keras.layers.Conv2D(64, (3, 3), activation=partial_linear),
        # keras.layers.MaxPooling2D((2, 2)),

        # Flattening Layer
        keras.layers.Flatten(),

        # First Fully Connected Layer
        keras.layers.Dense(64, activation='relu'),

        # Output Layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

    # Evaluate the model
    _, test_acc = model.evaluate(x_train, y_train, verbose=0)
    print('Test accuracy:', test_acc)

    # Print the summary of the model
    model.summary()


def lenet(x_train, y_train):
    # Implementation of the LeNet-5 architecture
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        # First Convolutional Layer
        keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Second Convolutional Layer
        keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Flattening Layer
        keras.layers.Flatten(),
    
        # First Fully Connected Layer
        keras.layers.Dense(units=120, activation='relu'),
    
        # Second Fully Connected Layer
        keras.layers.Dense(units=84, activation='relu'),
    
        # Output Layer
        keras.layers.Dense(units=10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

    # Evaluate the model
    _, test_acc = model.evaluate(x_train, y_train, verbose=0)
    print('Test accuracy:', test_acc)

    # Print the summary of the model
    model.summary()


def second_task(x_train, y_train):
    # Define the model architecture for the 2nd task
    model = models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        # First pooling layer
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Second pooling layer
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Flatten the output from the pooling layers
        keras.layers.Flatten(),
    
        # Dense layer for classification
        keras.layers.Dense(units=128, activation='relu'),
    
        # Output Layer
        keras.layers.Dense(units=10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

    # Evaluate the model
    _, test_acc = model.evaluate(x_train, y_train, verbose=0)
    print('Test accuracy:', test_acc)

    # Print the summary of the model
    model.summary()


if __name__ == '__main__':
    print('Выберете размер выборки (1 - small, иначе - large):')
    DATA_DIR = SMALL_DATA_DIR if input() == '1' else LARGE_DATA_DIR
    # Load the data
    x_train, y_train = load_data()

    # Normalize the images
    x_train = normalize(x_train)

    conv_neural_network(x_train, y_train)
    second_task(x_train, y_train)
    lenet(x_train, y_train)
