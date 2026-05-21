"""
Модуль для загрузки и управления моделью распознавания последовательности цифр.

Модель обучена на датасете SVHN (Street View House Numbers).
Архитектура: CNN → выход (5, 11) — 5 позиций цифр, 11 классов (0-9 + заглушка).
Веса сохранены в формате H5 через model.save_weights().
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import config

logger = logging.getLogger(__name__)


def _build_svhn_model() -> tf.keras.Model:
    """
    Пересоздать архитектуру модели SVHN из кода.

    Точная копия архитектуры из 2/lab2.py (build_svhn_model).
    Вход:  (32, 32, 1)  — grayscale изображение
    Выход: (5, 11)      — 5 позиций цифр, 11 классов каждая
    """
    layers = tf.keras.layers

    img_height = config.IMAGE_SIZE[0]   # 32
    img_width = config.IMAGE_SIZE[1]    # 32
    num_channels = config.IMAGE_CHANNELS  # 1
    num_digits = config.NUM_DIGIT_POSITIONS  # 5
    num_labels = config.NUM_CLASSES          # 11

    dropout_rate_conv = 0.50
    dropout_rate_fc = 0.50

    keras_init = 'glorot_uniform'  # xavier

    inputs = tf.keras.Input(
        shape=(img_height, img_width, num_channels), name='x'
    )

    # Conv Block 1
    x = layers.Conv2D(
        32, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)

    x = layers.Conv2D(
        32, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate_conv)(x)

    # Conv Block 2
    x = layers.Conv2D(
        64, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)

    x = layers.Conv2D(
        64, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_4'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate_conv)(x)

    # Conv Block 3
    x = layers.Conv2D(
        128, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_5'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)

    x = layers.Conv2D(
        128, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_6'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)

    x = layers.Conv2D(
        128, (5, 5), padding='same',
        kernel_initializer=keras_init, name='conv_7'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate_fc)(x)

    # Flatten
    x = layers.Flatten()(x)

    # FC 1
    x = layers.Dense(
        256, kernel_initializer=keras_init, name='fc_1'
    )(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)
    x = layers.Dropout(dropout_rate_fc)(x)

    # FC 2
    x = layers.Dense(
        256, kernel_initializer=keras_init, name='fc_2'
    )(x)
    x = layers.LeakyReLU(negative_slope=0.10)(x)

    # Выходной слой: (5 * 11) → reshape → (5, 11)
    outputs = layers.Dense(
        num_digits * num_labels, kernel_initializer=keras_init
    )(x)
    outputs = layers.Reshape(
        (num_digits, num_labels), name='y_pred'
    )(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SVHN_Model')
    return model


class ModelLoader:
    """Класс для загрузки и кэширования модели (Singleton)"""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[tf.keras.Model] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка модели: пересоздаём архитектуру SVHN и загружаем веса из H5.
        """
        try:
            logger.info(f"Загрузка модели из {config.MODEL_PATH}")

            if not os.path.exists(config.MODEL_PATH):
                raise FileNotFoundError(
                    f"Файл модели не найден: {config.MODEL_PATH}"
                )

            model = _build_svhn_model()
            logger.info("Архитектура модели SVHN создана")

            model.load_weights(config.MODEL_PATH)
            logger.info("Веса модели загружены из H5")

            self._model = model

            logger.info("Модель успешно загружена")
            logger.info(f"Входная форма: {self._model.input_shape}")
            logger.info(f"Выходная форма: {self._model.output_shape}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def get_model(self) -> tf.keras.Model:
        """Получить загруженную модель"""
        if self._model is None:
            self._load_model()
        return self._model

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Выполнить предсказание последовательности цифр для изображения.

        Args:
            image: Предобработанное изображение формы (32, 32, 1)

        Returns:
            Кортеж (digits, probabilities_per_position):
              - digits: список распознанных цифр (без заглушек, класс 10)
              - probabilities_per_position: список из 5 векторов вероятностей
                по 11 элементов каждый (логиты → softmax)
        """
        try:
            model = self.get_model()

            # (H, W, C) → (1, H, W, C)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)

            # Предсказание: логиты формы (1, 5, 11)
            logits = model.predict(image, verbose=0)  # (1, 5, 11)
            logits = logits[0]  # (5, 11)

            # Softmax по последней оси для получения вероятностей
            probs_all = tf.nn.softmax(logits, axis=-1).numpy()  # (5, 11)

            # Индексы предсказанных классов для каждой позиции
            predicted_classes = np.argmax(probs_all, axis=-1)  # (5,)

            # Отбираем только реальные цифры (не заглушки, класс != 10)
            digits = []
            probabilities_per_position = []
            for pos in range(config.NUM_DIGIT_POSITIONS):
                cls = int(predicted_classes[pos])
                probs = probs_all[pos].tolist()  # 11 вероятностей
                probabilities_per_position.append(probs)
                if cls != config.BLANK_CLASS:
                    digits.append(cls)

            return digits, probabilities_per_position

        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {e}")
            raise

    def get_model_info(self) -> Dict:
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
