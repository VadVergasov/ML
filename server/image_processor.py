"""
Модуль для предобработки изображений перед подачей в модель
"""

import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

import config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс для предобработки изображений"""
    
    def __init__(self):
        """Инициализация процессора изображений"""
        self.target_size = config.IMAGE_SIZE
        self.channels = config.IMAGE_CHANNELS
        self.norm_factor = config.NORMALIZATION_FACTOR
    
    def process_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Предобработать изображение из байтов
        
        Args:
            image_bytes: Байты изображения (PNG, JPEG и т.д.)
            
        Returns:
            Предобработанное изображение в формате numpy array формы (H, W, C)
        """
        try:
            # Декодируем изображение из байтов
            image = Image.open(BytesIO(image_bytes))
            
            # Конвертируем в RGB, если изображение в другом формате
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Конвертируем в numpy array
            image_array = np.array(image)
            
            # Предобрабатываем
            processed = self._preprocess(image_array)
            
            logger.debug(
                f"Изображение успешно предобработано: {processed.shape}"
            )
            return processed
            
        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}")
            raise
    
    def process_from_file(self, file_path: str) -> np.ndarray:
        """
        Предобработать изображение из файла
        
        Args:
            file_path: Путь к файлу изображения
            
        Returns:
            Предобработанное изображение в формате numpy array формы (H, W, C)
        """
        try:
            # Загружаем изображение
            image = Image.open(file_path)
            
            # Конвертируем в RGB, если изображение в другом формате
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Конвертируем в numpy array
            image_array = np.array(image)
            
            # Предобрабатываем
            processed = self._preprocess(image_array)
            
            logger.debug(
                f"Изображение из файла успешно предобработано: "
                f"{processed.shape}"
            )
            return processed
            
        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения из файла: {e}")
            raise
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Внутренний метод предобработки
        
        Args:
            image: Изображение в формате numpy array (H, W, C)
            
        Returns:
            Предобработанное изображение формы (H, W, 1) для модели
        """
        # Добавляем размерность батча, если её нет
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Конвертируем в float32
        image = image.astype(np.float32)
        
        # Нормализуем в диапазон [0, 1]
        image = image / self.norm_factor
        
        # Конвертируем в grayscale (как при обучении модели)
        # Используем tf.image.rgb_to_grayscale для совместимости с обучением
        if image.shape[-1] == 3:
            # Добавляем размерность батча для TensorFlow
            image_batch = np.expand_dims(image, axis=0)
            grayscale = tf.image.rgb_to_grayscale(image_batch)
            image = grayscale.numpy()[0]
        
        # Resize до целевого размера (как при обучении: tf.image.resize)
        if image.shape[:2] != self.target_size:
            # Добавляем размерность батча для TensorFlow
            image_batch = np.expand_dims(image, axis=0)
            resized = tf.image.resize(image_batch, self.target_size)
            image = resized.numpy()[0]
        
        # Убеждаемся, что форма правильная (H, W, 1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def validate_image(self, image_bytes: bytes) -> bool:
        """
        Проверить, что байты содержат валидное изображение
        
        Args:
            image_bytes: Байты для проверки
            
        Returns:
            True, если это валидное изображение, иначе False
        """
        try:
            Image.open(BytesIO(image_bytes))
            return True
        except Exception:
            return False
    
    def get_image_info(self, image_bytes: bytes) -> dict:
        """
        Получить информацию об изображении
        
        Args:
            image_bytes: Байты изображения
            
        Returns:
            Словарь с информацией об изображении
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
        except Exception as e:
            logger.error(
                f"Ошибка при получении информации об изображении: {e}"
            )
            return {}
