"""
Модуль для загрузки и управления моделью распознавания цифр.

Модель сохранена через tensorflow.keras в формате H5.
Из-за несовместимости версий Keras пересоздаём архитектуру
из кода и загружаем только веса из H5 файла.
"""

import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf

import config

logger = logging.getLogger(__name__)


def _build_model():
    """
    Пересоздать архитектуру модели из кода.

    Архитектура взята из 2/lab2.py — точная копия модели,
    которая была обучена и сохранена в numbers.h5.
    """
    layers = tf.keras.layers

    def residual_block(x, filters, downsample=False):
        shortcut = x
        strides = 2 if downsample else 1

        x = layers.Conv2D(
            filters, (3, 3), strides=strides, padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        if downsample or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv2D(
                filters, (1, 1), strides=strides, padding="same"
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    def residual_block_light(x, filters):
        shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
        x = layers.Conv2D(
            filters, (3, 3), padding="same", activation="relu"
        )(x)
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = residual_block_light(x, 16)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model


class ModelLoader:
    """Класс для загрузки и кэширования модели (Singleton)"""

    _instance: Optional['ModelLoader'] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка модели: пересоздаём архитектуру и загружаем веса из H5.

        Это обходит проблему несовместимости конфигурации слоёв
        между разными версиями Keras.
        """
        try:
            logger.info(f"Загрузка модели из {config.MODEL_PATH}")

            if not os.path.exists(config.MODEL_PATH):
                raise FileNotFoundError(
                    f"Файл модели не найден: {config.MODEL_PATH}"
                )

            # Пересоздаём архитектуру из кода
            model = _build_model()
            logger.info("Архитектура модели создана")

            # Загружаем только веса из H5 файла
            model.load_weights(config.MODEL_PATH)
            logger.info("Веса модели загружены из H5")

            self._model = model

            logger.info("Модель успешно загружена")
            logger.info(f"Входная форма: {self._model.input_shape}")
            logger.info(f"Выходная форма: {self._model.output_shape}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def get_model(self):
        """Получить загруженную модель"""
        if self._model is None:
            self._load_model()
        return self._model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Выполнить предсказание для изображения

        Args:
            image: Предобработанное изображение формы (H, W, C)

        Returns:
            Вектор вероятностей для каждого из 10 классов
        """
        try:
            model = self.get_model()

            # Добавляем размерность батча: (H, W, C) -> (1, H, W, C)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)

            predictions = model.predict(image, verbose=0)
            return predictions[0]

        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {e}")
            raise

    def get_model_info(self) -> dict:
        """Получить информацию о модели"""
        model = self.get_model()
        return {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_params': model.count_params(),
            'model_path': config.MODEL_PATH
        }

    def reload_model(self) -> None:
        """Перезагрузить модель из файла"""
        logger.info("Перезагрузка модели...")
        self._model = None
        self._load_model()
        logger.info("Модель успешно перезагружена")
