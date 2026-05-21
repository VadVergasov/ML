"""
Flask-приложение для сервера распознавания последовательности цифр (SVHN).

Модель принимает всё изображение целиком и предсказывает
до 5 цифр сразу (выход формы (5, 11)).
"""

import logging
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from image_processor import ImageProcessor
from model_loader import ModelLoader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание Flask-приложения
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Настройка CORS
CORS(app, origins=config.CORS_ORIGINS)

# Инициализация компонентов
model_loader = ModelLoader()
image_processor = ImageProcessor()


def allowed_file(filename: str) -> bool:
    """Проверить что расширение файла разрешено"""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Проверка работоспособности сервера.

    Returns:
        JSON с информацией о статусе сервера и модели
    """
    try:
        model_info = model_loader.get_model_info()
        return jsonify({
            'status': 'healthy',
            'model': {
                'input_shape': [
                    str(d) if d is not None else None
                    for d in model_info['input_shape']
                ],
                'output_shape': [
                    str(d) if d is not None else None
                    for d in model_info['output_shape']
                ],
                'num_params': model_info['num_params'],
                'model_path': model_info['model_path']
            },
            'config': {
                'image_size': config.IMAGE_SIZE,
                'image_channels': config.IMAGE_CHANNELS,
                'allowed_extensions': list(config.ALLOWED_EXTENSIONS)
            }
        }), 200
    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Распознавание последовательности цифр на изображении.

    Принимает multipart/form-data с полем 'image'.
    Модель предсказывает до 5 цифр сразу из всего изображения.

    Returns:
        JSON:
        {
            "success": true,
            "digits": [
                {
                    "digit": 4,
                    "confidence": 0.97,
                    "probabilities": [0.0, ..., 0.97, ...]  // 11 значений
                },
                ...
            ],
            "number": "42",
            "digits_count": 2
        }
    """
    try:
        # Проверяем наличие файла
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Отсутствует файл изображения'
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Файл не выбран'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': (
                    f'Недопустимый формат. Разрешены: '
                    f'{", ".join(config.ALLOWED_EXTENSIONS)}'
                )
            }), 400

        image_bytes = file.read()

        if not image_processor.validate_image(image_bytes):
            return jsonify({
                'success': False,
                'error': 'Невалидное изображение'
            }), 400

        # Предобрабатываем всё изображение целиком → (32, 32, 1)
        processed_image = image_processor.preprocess(image_bytes)

        # Предсказание: digits — список цифр без заглушек,
        # probs_per_pos — список из 5 векторов по 11 вероятностей
        digits, probs_per_pos = model_loader.predict(processed_image)

        # Формируем детальный ответ по каждой позиции
        results = []
        for pos_idx, probs in enumerate(probs_per_pos):
            # Индекс предсказанного класса для этой позиции
            predicted_cls = int(max(range(len(probs)), key=lambda i: probs[i]))

            # Пропускаем позиции-заглушки (класс 10)
            if predicted_cls == config.BLANK_CLASS:
                continue

            confidence = float(probs[predicted_cls])
            results.append({
                'digit': predicted_cls,
                'confidence': confidence,
                # Передаём все 11 вероятностей (0-9 + заглушка)
                'probabilities': [float(p) for p in probs]
            })

        number_str = ''.join(str(d['digit']) for d in results)

        logger.info(
            f"Распознан номер: '{number_str}' "
            f"({len(results)} цифр)"
        )

        return jsonify({
            'success': True,
            'digits': results,
            'number': number_str,
            'digits_count': len(results)
        }), 200

    except Exception as e:
        logger.error(f"Ошибка при распознавании: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Ошибка при обработке: {str(e)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error) -> Dict[str, Any]:
    """Файл слишком большой"""
    max_mb = config.MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify({
        'success': False,
        'error': f'Файл слишком большой. Максимум: {max_mb:.0f} MB'
    }), 413


@app.errorhandler(500)
def internal_server_error(error) -> Dict[str, Any]:
    """Внутренняя ошибка сервера"""
    logger.error(f"Внутренняя ошибка: {error}")
    return jsonify({
        'success': False,
        'error': 'Внутренняя ошибка сервера'
    }), 500


def main():
    """Запуск сервера"""
    logger.info("Запуск сервера распознавания цифр (SVHN)...")
    logger.info(f"Хост: {config.HOST}, Порт: {config.PORT}")
    logger.info(f"Путь к модели: {config.MODEL_PATH}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == '__main__':
    main()
