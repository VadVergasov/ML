"""
Flask-приложение для сервера распознавания цифр
"""

import logging
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

import config
from model_loader import ModelLoader
from image_processor import ImageProcessor

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
    """
    Проверить, что расширение файла разрешено
    
    Args:
        filename: Имя файла
        
    Returns:
        True, если расширение разрешено, иначе False
    """
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Эндпоинт для проверки работоспособности сервера
    
    Returns:
        JSON с информацией о статусе сервера и модели
    """
    try:
        model_info = model_loader.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'model': {
                'input_shape': [
                    str(dim) if dim is not None else None
                    for dim in model_info['input_shape']
                ],
                'output_shape': [
                    str(dim) if dim is not None else None
                    for dim in model_info['output_shape']
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
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Эндпоинт для распознавания цифры на изображении
    
    Ожидает multipart/form-data с полем 'image', содержащим файл изображения.
    
    Returns:
        JSON с результатами распознавания:
        {
            "success": true,
            "predictions": [0.01, 0.02, 0.85, ...],
            "predicted_class": 2,
            "confidence": 0.85
        }
    """
    try:
        # Проверяем наличие файла в запросе
        if 'image' not in request.files:
            logger.warning("Запрос без файла изображения")
            return jsonify({
                'success': False,
                'error': 'Отсутствует файл изображения'
            }), 400
        
        file = request.files['image']
        
        # Проверяем, что файл выбран
        if file.filename == '':
            logger.warning("Пустое имя файла")
            return jsonify({
                'success': False,
                'error': 'Файл не выбран'
            }), 400
        
        # Проверяем расширение файла
        if not allowed_file(file.filename):
            logger.warning(f"Недопустимое расширение файла: {file.filename}")
            return jsonify({
                'success': False,
                'error': (
                    f'Недопустимый формат файла. Разрешены: '
                    f'{", ".join(config.ALLOWED_EXTENSIONS)}'
                )
            }), 400
        
        # Читаем байты изображения
        image_bytes = file.read()
        
        # Проверяем, что это валидное изображение
        if not image_processor.validate_image(image_bytes):
            logger.warning("Невалидное изображение")
            return jsonify({
                'success': False,
                'error': 'Невалидное изображение'
            }), 400
        
        # Предобрабатываем изображение
        processed_image = image_processor.process_from_bytes(image_bytes)
        
        # Выполняем предсказание
        predictions = model_loader.predict(processed_image)
        
        # Находим предсказанный класс и уверенность
        predicted_class = int(predictions.argmax())
        confidence = float(predictions[predicted_class])
        
        # Формируем ответ
        response = {
            'success': True,
            'predictions': [float(p) for p in predictions],
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        logger.info(
            f"Распознавание успешно: класс={predicted_class}, "
            f"уверенность={confidence:.4f}"
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Ошибка при распознавании: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Ошибка при обработке изображения: {str(e)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error) -> Dict[str, Any]:
    """Обработка ошибки слишком большого файла"""
    return jsonify({
        'success': False,
        'error': (
            f'Файл слишком большой. Максимальный размер: '
            f'{config.MAX_CONTENT_LENGTH / (1024 * 1024):.0f} MB'
        )
    }), 413


@app.errorhandler(500)
def internal_server_error(error) -> Dict[str, Any]:
    """Обработка внутренней ошибки сервера"""
    logger.error(f"Внутренняя ошибка сервера: {error}")
    return jsonify({
        'success': False,
        'error': 'Внутренняя ошибка сервера'
    }), 500


def main():
    """Запуск сервера"""
    logger.info("Запуск сервера распознавания цифр...")
    logger.info(f"Хост: {config.HOST}, Порт: {config.PORT}")
    logger.info(f"Путь к модели: {config.MODEL_PATH}")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )


if __name__ == '__main__':
    main()
