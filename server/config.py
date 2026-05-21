"""
Конфигурация сервера распознавания цифр
"""

import os
from pathlib import Path

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# Путь к модели H5
MODEL_PATH = str(BASE_DIR / "2" / "svhn_model.weights.h5")

# Настройки сервера
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8888"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Настройки изображений
IMAGE_SIZE = (32, 32)  # Размер, на котором обучалась модель (SVHN)
IMAGE_CHANNELS = 1     # Grayscale

# Параметры нормализации SVHN (grayscale, вычислены на тренировочном наборе)
# Модель обучена с нормализацией: (x - mean) / std
# Используем стандартные значения для grayscale SVHN
# Если точные значения неизвестны — используем приближение ImageNet grayscale
NORMALIZE_MEAN = 0.4377  # среднее по grayscale SVHN train
NORMALIZE_STD = 0.1980   # std по grayscale SVHN train

# Максимальный размер загружаемого файла (в байтах)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Разрешённые форматы изображений
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}

# CORS настройки
CORS_ORIGINS = ["*"]  # Разрешить все источники для разработки

# Параметры модели SVHN
NUM_DIGIT_POSITIONS = 5   # Модель предсказывает до 5 цифр
NUM_CLASSES = 11          # 0-9 + заглушка (10 = пустая позиция)
BLANK_CLASS = 10          # Индекс класса «нет цифры»
