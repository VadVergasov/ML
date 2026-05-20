"""
Конфигурация сервера распознавания цифр
"""

import os
from pathlib import Path

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# Путь к модели H5
MODEL_PATH = str(BASE_DIR / "2" / "numbers.h5")

# Настройки сервера
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8888"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Настройки изображений
IMAGE_SIZE = (28, 28)  # Размер, на котором обучалась модель
IMAGE_CHANNELS = 1  # Grayscale
NORMALIZATION_FACTOR = 255.0  # Делить на это значение для нормализации

# Максимальный размер загружаемого файла (в байтах)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Разрешённые форматы изображений
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}

# CORS настройки
CORS_ORIGINS = ["*"]  # Разрешить все источники для разработки
