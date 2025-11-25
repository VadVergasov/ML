from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import numpy as np
import os
import random


LARGE_DATA_DIR = 'notMNIST_large'
SMALL_DATA_DIR = 'notMNIST_small'

print('Выберете размер выборки (1 - small, иначе - large):')

DATA_DIR = SMALL_DATA_DIR if input() == '1' else LARGE_DATA_DIR
LETTERS = 'ABCDEFGHIJ'

image_data = []
image_labels = []

def load_images_with_progress(image_dir, class_label, letter):
    image_data = []
    image_labels = []
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    for filename in tqdm(files, desc=f"Загрузка {class_label}", unit="img"):
        path = os.path.join(image_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_data.append(img)
            image_labels.append(ord(letter) - ord('A'))

    return image_data, image_labels

for letter in LETTERS:
    folder_path = os.path.join(os.path.expanduser("~"), DATA_DIR, letter)
    img_data, img_labels = load_images_with_progress(folder_path, letter, letter)
    image_data.extend(img_data)
    image_labels.extend(img_labels)

image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(f"Загружено {len(image_data)} изображений")
print(f"Классы: {np.unique(image_labels)}")

TOTAL_IMAGES = len(image_data)

# Разделим на обучающую и оставшуюся (валидация + тест)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    image_data, image_labels, test_size=0.2, random_state=42
)

# Разделим оставшуюся на валидационную и тестовую
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.2, random_state=42
)

print(f"Обучающая выборка: {len(train_data)}")
print(f"Валидационная выборка: {len(val_data)}")
print(f"Тестовая выборка: {len(test_data)}")

def remove_duplicates(train_data, train_labels):
    seen_images = set()
    unique_indices = []
    
    for i in range(len(train_data)):
        img_flat = tuple(train_data[i].flatten())
        if img_flat not in seen_images:
            seen_images.add(img_flat)
            unique_indices.append(i)
    
    unique_train_data = train_data[unique_indices]
    unique_train_labels = train_labels[unique_indices]
    
    print(f"Удалено {len(train_data) - len(unique_train_data)} дубликатов")
    return unique_train_data, unique_train_labels

train_data, train_labels = remove_duplicates(train_data, train_labels)
print(f"Обновленная обучающая выборка: {len(train_data)}")

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# --- Гиперпараметры
BATCH_SIZE = 8192
EPOCHS = 100
LEARNING_RATE = 0.1
DROPOUT_RATE = 0.5
L2_REG = 0.01
DECAY_RATE = 0.1

# --- Функция для обучения с динамическим LR
def train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LEARNING_RATE):
    model = models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(128, activation='silu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
        layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(10),
    ])

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1000,
        decay_rate=DECAY_RATE,
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Обучение
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model

# --- Функция для оценки на тесте
def evaluate_model(model, X_test, y_test):
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc

model = train_model(train_data, train_labels, val_data, val_labels)
test_acc = evaluate_model(model, test_data, test_labels)
print(f"Точность на тесте: {test_acc:.4f}")

def train_and_evaluate(X_train, y_train, X_val, y_val):
    X_train_sample = np.array([row.flatten() for row in X_train[:50000]])
    y_train_sample = y_train[:50000]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_val_scaled = scaler.transform(np.array([row.flatten() for row in X_val]))
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train_sample)
    
    y_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    return acc

lr_acc = train_and_evaluate(train_data, train_labels, val_data, val_labels)
