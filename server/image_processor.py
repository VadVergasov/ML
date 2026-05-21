"""
Модуль для предобработки изображений перед распознаванием цифр.

Модель SVHN принимает всё изображение целиком (32x32, grayscale).
Предобработка повторяет шаги из 2/lab2.py:
  1. RGB -> grayscale
  2. resize до 32x32
  3. нормализация: (x - mean) / std
"""

import logging
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

import config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс для предобработки изображений перед подачей в модель SVHN"""

    def __init__(self):
        self.target_size = config.IMAGE_SIZE      # (32, 32)
        self.mean = config.NORMALIZE_MEAN
        self.std = config.NORMALIZE_STD

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Предобработать изображение для подачи в модель SVHN.

        Шаги (повторяют 2/lab2.py):
          1. Открыть изображение, привести к RGB
          2. float32, нормализация [0, 1] делением на 255
          3. rgb_to_grayscale
          4. resize до (32, 32)
          5. нормализация: (x - mean) / std

        Args:
            image_bytes: Байты изображения

        Returns:
            numpy array формы (32, 32, 1)
        """
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            img_rgb = np.array(pil_image, dtype=np.float32)

            return self._preprocess_array(img_rgb)

        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}")
            raise

    def validate_image(self, image_bytes: bytes) -> bool:
        """Проверить что байты содержат валидное изображение"""
        try:
            Image.open(BytesIO(image_bytes))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _preprocess_array(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Предобработать RGB numpy array до формата модели (32, 32, 1).

        Args:
            img_rgb: float32 RGB массив любого размера

        Returns:
            numpy array формы (32, 32, 1), нормализованный
        """
        # Убеждаемся что 3 канала
        if len(img_rgb.shape) == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        elif img_rgb.shape[-1] == 4:
            img_rgb = img_rgb[:, :, :3]

        # float32 + нормализация [0, 1]
        img = img_rgb.astype(np.float32) / 255.0

        # rgb_to_grayscale через tf (как при обучении)
        img_batch = np.expand_dims(img, axis=0)          # (1, H, W, 3)
        gray = tf.image.rgb_to_grayscale(img_batch)      # (1, H, W, 1)

        # resize до 32x32 через tf (как при обучении)
        resized = tf.image.resize(
            gray, list(self.target_size)
        )                                                 # (1, 32, 32, 1)

        result = resized.numpy()[0]                       # (32, 32, 1)

        # Нормализация: (x - mean) / std  (как при обучении SVHN)
        result = (result - self.mean) / self.std

        return result.astype(np.float32)
