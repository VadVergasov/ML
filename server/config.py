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
NORMALIZE_MEAN = 0.4377  # среднее по grayscale SVHN train
NORMALIZE_STD = 0.1980   # std по grayscale SVHN train

# Максимальный размер загружаемого файла (в байтах)
MAX_CONTENT_LENGTH = 256 * 1024 * 1024  # 256 MB

# Разрешённые форматы изображений
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}

# CORS настройки
CORS_ORIGINS = ["*"]  # Разрешить все источники для разработки

# Параметры модели SVHN
NUM_DIGIT_POSITIONS = 5   # Модель предсказывает до 5 цифр
NUM_CLASSES = 11          # 0-9 + заглушка (10 = пустая позиция)
BLANK_CLASS = 10          # Индекс класса «нет цифры»

# Отступ вокруг найденного bbox детектором (в пикселях)
DETECTOR_BBOX_PADDING = 10

# Настройки MSER детектора текстовых регионов (OpenCV)
# delta — чувствительность MSER (меньше = больше регионов, 5 оптимально)
MSER_DELTA = int(os.getenv("MSER_DELTA", "5"))
# Минимальная и максимальная площадь региона в пикселях
# Маленькое min_area позволяет находить мелкие цифры
MSER_MIN_AREA = int(os.getenv("MSER_MIN_AREA", "30"))
MSER_MAX_AREA = int(os.getenv("MSER_MAX_AREA", "14400"))
# Минимальная высота региона как доля высоты изображения
# 0.03 = 3% — позволяет находить небольшие цифры
MSER_MIN_HEIGHT_RATIO = float(os.getenv("MSER_MIN_HEIGHT_RATIO", "0.05"))
# Максимальная высота региона как доля высоты изображения
MSER_MAX_HEIGHT_RATIO = float(os.getenv("MSER_MAX_HEIGHT_RATIO", "0.90"))
# Минимальная ширина региона как доля ширины изображения
# 0.01 = 1% — позволяет находить узкие цифры (1, i)
MSER_MIN_WIDTH_RATIO = float(os.getenv("MSER_MIN_WIDTH_RATIO", "0.02"))
# Максимальное соотношение ширина/высота (цифры не слишком широкие)
MSER_MAX_ASPECT_RATIO = float(os.getenv("MSER_MAX_ASPECT_RATIO", "3.0"))
# Максимальный горизонтальный зазор между регионами одного кластера
# Увеличен до 30px чтобы объединять цифры одного числа
MSER_CLUSTER_GAP = int(os.getenv("MSER_CLUSTER_GAP", "30"))
# Максимальный размер итогового bbox как доля размера изображения.
# Ограничивает выбор слишком больших областей (фон, весь кадр).
MSER_MAX_BBOX_WIDTH_RATIO = float(
    os.getenv("MSER_MAX_BBOX_WIDTH_RATIO", "0.85")
)
MSER_MAX_BBOX_HEIGHT_RATIO = float(
    os.getenv("MSER_MAX_BBOX_HEIGHT_RATIO", "0.85")
)
# Минимальное количество MSER-регионов в группе (строке).
# Группы с меньшим числом регионов считаются шумом.
# 2 = минимум 2 региона (одна цифра может дать 2+ региона)
MSER_MIN_GROUP_SIZE = int(os.getenv("MSER_MIN_GROUP_SIZE", "2"))

