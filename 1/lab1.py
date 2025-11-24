from collections import Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
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

def load_images_with_progress(image_dir, class_label, class_index):
    image_data = []
    image_labels = []
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    for filename in tqdm(files, desc=f"Загрузка {class_label}", unit="img"):
        path = os.path.join(image_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_data.append(img.flatten())
            image_labels.append(class_index)

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

random_indices = random.sample(range(len(image_data)), 10)

# Отображаем первые 10 изображений
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i, idx in enumerate(random_indices):
    img = image_data[idx].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Класс: {image_labels[idx]}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Подсчёт количества изображений по классам
class_counts = Counter(image_labels)
print("Количество изображений по классам:")
for letter, count in class_counts.items():
    print(f"  {letter}: {count}")

# Проверка баланса
total_images = len(image_data)
expected_per_class = total_images // len(LETTERS)
print(f"Ожидаемое количество на класс: {expected_per_class}")
print(f"Фактическое среднее: {np.mean(list(class_counts.values())):.1f}")
print(f"Максимальное отклонение: {max(abs(count - expected_per_class) for count in class_counts.values())}")

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

# Функция для обучения и оценки точности
def train_and_evaluate(X_train, y_train, X_val, y_val, sample_size):
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_val_scaled = scaler.transform(X_val)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train_sample)
    
    y_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    return acc

sample_sizes = [50, 100, 1000, 50000]
accuracies = []

for size in sample_sizes:
    acc = train_and_evaluate(train_data, train_labels, val_data, val_labels, size)
    accuracies.append(acc)
    print(f"Размер выборки: {size} → Точность: {acc:.4f}")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, accuracies, marker='o', color='b', linewidth=2, label='Точность (LogReg)')
plt.title('Зависимость точности от размера обучающей выборки')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность')
plt.grid(True)
plt.xscale('log')
plt.legend()
plt.show()
