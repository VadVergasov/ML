from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from scipy.io import loadmat


def residual_block(x, filters, downsample=False):
    shortcut = x
    strides = 2 if downsample else 1

    x = layers.Conv2D(filters, (3, 3), strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    if downsample or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def residual_block_light(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


inputs = keras.Input(shape=(28, 28, 1))  # 28×28 черно-белые изображения
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
x = residual_block_light(x, 16)
x = layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 32)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(10, activation="softmax")(x)  # 10 классов цифр

model = keras.Model(inputs, x)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

train = loadmat('/home/vadzim/train_32x32.mat')

print(len(train))

x_train = np.transpose(train['X'], (3, 0, 1, 2)).astype(np.float32) / 255.0

y_train = train['y'].astype(np.uint8).flatten()

# т.к. цифра 0 хранится как 10
y_train[y_train == 10] = 0

test = loadmat('/home/vadzim/test_32x32.mat')

x_test = np.transpose(test['X'], (3, 0, 1, 2)).astype(np.float32) / 255.0

y_test = test['y'].astype(np.uint8).flatten()
y_test[y_test == 10] = 0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = tf.image.rgb_to_grayscale(x_train).numpy()
x_test = tf.image.rgb_to_grayscale(x_test).numpy()

x_train = tf.image.resize(x_train, [28, 28]).numpy()
x_test = tf.image.resize(x_test, [28, 28]).numpy()

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Точность модели: {test_acc * 100:.2f}%\nПотери: {test_loss}')

def show_samples(x, y, n=10):
    idxs = np.random.choice(len(x), n, replace=False)
    
    plt.figure(figsize=(12, 3))
    
    for i, idx in enumerate(idxs):
        plt.subplot(1, n, i+1)
        plt.imshow(x[idx], cmap='gray')
        plt.title(int(y[idx]))
        plt.axis('off')
    
    plt.show()

show_samples(x_train, y_train, n=10)

def predict_image_and_show(model, count):
    idxs = np.random.choice(len(x_train), count, replace=False)

    for i, idx in enumerate(idxs):
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 1, i+1)
        plt.imshow(x_train[idx], cmap='gray')
        plt.title(int(y_train[idx]))
        plt.axis('off')
        plt.show()
        preds = model.predict(idx, verbose=0)
        print(preds)

# predict_image_and_show(model, 10)

model.save("numbers.h5")
