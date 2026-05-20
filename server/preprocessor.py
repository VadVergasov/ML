"""
Модуль для предобработки изображений перед распознаванием
"""

import io
import logging

import numpy as np
import tensorflow as tf
from PIL import Image

import config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Класс для предобработки изображений"""

    def __init__(self):
        """Инициализация препроцессора"""
        self.target_size = config.IMAGE_SIZE

    def preprocess_from_file(self, file_path: str) -> np.ndarray:
        """
        Предобработать изображение из файла

        Args:
            file_path: Путь к файлу изображения

        Returns:
            Предобработанное изображение в формате numpy array
        """
        try:
            image = Image.open(file_path)
            return self._preprocess_image(image)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {e}")
            raise

    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Предобработать изображение из байтов

        Args:
            image_bytes: Байты изображения

        Returns:
            Предобработанное изображение в формате numpy array
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self._preprocess_image(image)
        except Exception as e:
            logger.error(
                f"Ошибка при обработке изображения из байтов: {e}"
            )
            raise

    def preprocess_from_storage(self, storage) -> np.ndarray:
        """
        Предобработать изображение из объекта хранения Flask

        Args:
            storage: Объект хранения файла из Flask request.files

        Returns:
            Предобработанное изображение в формате numpy array
        """
        try:
            image_bytes = storage.read()
            return self.preprocess_from_bytes(image_bytes)
        except Exception as e:
            logger.error(
                f"Ошибка при обработке изображения из storage: {e}"
            )
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Внутренний метод предобработки изображения.

        Повторяет шаги обучения модели:
        1. Конвертация в RGB
        2. Нормализация [0, 1]
        3. tf.image.rgb_to_grayscale (как при обучении)
        4. tf.image.resize до 28x28 (как при обучении)

        Args:
            image: Объект PIL Image

        Returns:
            Предобработанное изображение формы (28, 28, 1)
        """
        try:
            # Конвертация в RGB, если изображение в другом формате
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Конвертация в numpy array и float32
            image_array = np.array(image, dtype=np.float32)

            # Нормализация значений пикселей [0, 255] -> [0, 1]
            image_array = image_array / config.NORMALIZATION_FACTOR

            # Конвертация в grayscale через tf (как при обучении)
            image_batch = np.expand_dims(image_array, axis=0)
            grayscale = tf.image.rgb_to_grayscale(image_batch)
            image_array = grayscale.numpy()[0]

            # Resize до целевого размера через tf (как при обучении)
            if image_array.shape[:2] != tuple(self.target_size):
                image_batch = np.expand_dims(image_array, axis=0)
                resized = tf.image.resize(image_batch, self.target_size)
                image_array = resized.numpy()[0]

            # Убеждаемся что форма (28, 28, 1)
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=-1)

            logger.debug(
                f"Форма предобработанного изображения: {image_array.shape}"
            )

            return image_array

        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}")
            raise

    def validate_image_size(self, image_bytes: bytes) -> bool:
        """
        Проверить размер изображения

        Args:
            image_bytes: Байты изображения

        Returns:
            True если размер допустим, иначе False
        """
        size = len(image_bytes)
        if size > config.MAX_CONTENT_LENGTH:
            logger.warning(
                f"Размер изображения {size} байт превышает "
                f"максимальный {config.MAX_CONTENT_LENGTH} байт"
            )
            return False
        return True

    def get_image_info(self, image: Image.Image) -> dict:
        """
        Получить информацию об изображении

        Args:
            image: Объект PIL Image

        Returns:
            Словарь с информацией об изображении
        """
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format
        }

    @staticmethod
    def is_valid_extension(filename: str) -> bool:
        """
        Проверить расширение файла

        Args:
            filename: Имя файла

        Returns:
            True если расширение допустимо, иначе False
        """
        if not filename:
            return False

        if '.' not in filename:
            return False
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in config.ALLOWED_EXTENSIONS
