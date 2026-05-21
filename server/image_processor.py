"""
Модуль для предобработки изображений перед распознаванием цифр.

Модель SVHN принимает всё изображение целиком (32x32, grayscale).
Предобработка повторяет шаги из 2/lab2.py:
  1. RGB -> grayscale
  2. resize до 32x32
  3. нормализация: (x - mean) / std

Если включён YOLOv8-детектор, перед предобработкой вырезается
bbox с наибольшей уверенностью как область с цифрами.
"""

import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import config

logger = logging.getLogger(__name__)

# Тип bbox: (x1, y1, x2, y2) в пикселях
BBox = Tuple[int, int, int, int]


class ImageProcessor:
    """Класс для предобработки изображений перед подачей в модель SVHN"""

    def __init__(self):
        self.target_size = config.IMAGE_SIZE      # (32, 32)
        self.mean = config.NORMALIZE_MEAN
        self.std = config.NORMALIZE_STD

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image_bytes: bytes,
        bbox: Optional[BBox] = None,
    ) -> np.ndarray:
        """
        Предобработать изображение для подачи в модель SVHN.

        Если передан bbox — вырезает только эту область (с отступом).
        Шаги (повторяют 2/lab2.py):
          1. Открыть изображение, привести к RGB
          2. Вырезать bbox (если задан)
          3. float32, нормализация [0, 1] делением на 255
          4. rgb_to_grayscale
          5. resize до (32, 32)
          6. нормализация: (x - mean) / std

        Args:
            image_bytes: Байты изображения
            bbox: Опциональный (x1, y1, x2, y2) для кропа

        Returns:
            numpy array формы (32, 32, 1)
        """
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            # Применяем EXIF-ориентацию (важно для JPEG с камеры)
            pil_image = ImageOps.exif_transpose(pil_image)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            img_rgb = np.array(pil_image, dtype=np.float32)

            # Кроп по bbox с отступом
            if bbox is not None:
                img_rgb = self._crop_with_padding(img_rgb, bbox)

            return self._preprocess_array(img_rgb)

        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}")
            raise

    def load_rgb(self, image_bytes: bytes) -> np.ndarray:
        """
        Загрузить изображение как RGB uint8 numpy array.

        Используется для передачи в YOLOv8 детектор.

        Args:
            image_bytes: Байты изображения

        Returns:
            numpy array формы (H, W, 3), dtype=uint8
        """
        pil_image = Image.open(BytesIO(image_bytes))
        # Применяем EXIF-ориентацию (важно для JPEG с камеры)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image, dtype=np.uint8)

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

    def _crop_with_padding(
        self, img_rgb: np.ndarray, bbox: BBox
    ) -> np.ndarray:
        """
        Вырезать область bbox с отступом YOLO_BBOX_PADDING.

        Args:
            img_rgb: float32 RGB массив (H, W, 3)
            bbox: (x1, y1, x2, y2) в пикселях

        Returns:
            Вырезанный фрагмент float32 RGB
        """
        h, w = img_rgb.shape[:2]
        pad = config.DETECTOR_BBOX_PADDING
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        cropped = img_rgb[y1:y2, x1:x2]

        # Защита от пустого кропа
        if cropped.size == 0:
            logger.warning("Пустой кроп по bbox, используем всё изображение")
            return img_rgb

        return cropped

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
