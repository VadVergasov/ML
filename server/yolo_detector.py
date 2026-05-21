"""
Модуль для детекции области с цифрами через YOLOv8.

Проблема: yolov8n обучена на COCO (80 классов), среди которых нет
класса "номер дома". Поэтому используем следующую стратегию:

1. Запускаем детекцию с низким порогом conf=0.05
2. Сначала ищем bbox среди "предпочтительных" классов COCO
   (знаки, часы, книги — объекты с текстом/цифрами)
3. Если не найдено — берём bbox по стратегии из config:
   - "center": ближайший к центру изображения
   - "confidence": с наибольшей уверенностью
4. Если YOLO вообще ничего не нашла — возвращаем None
   (используется всё изображение)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# Тип bbox: (x1, y1, x2, y2) в пикселях
BBox = Tuple[int, int, int, int]


class YOLODetector:
    """Singleton-детектор на базе YOLOv8n."""

    _instance: Optional['YOLODetector'] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Загрузить YOLOv8n с предобученными весами COCO."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            logger.info(
                f"Загрузка YOLOv8 из {config.YOLO_MODEL_PATH}"
            )
            self._model = YOLO(config.YOLO_MODEL_PATH)
            logger.info("YOLOv8 успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки YOLOv8: {e}")
            raise

    def detect(self, img_rgb: np.ndarray) -> Optional[BBox]:
        """
        Найти bbox области с цифрами на изображении.

        Стратегия:
          1. Детекция с низким порогом (conf=0.05)
          2. Приоритет — предпочтительные классы COCO (знаки, текст)
          3. Fallback — стратегия из config (center / confidence)

        Args:
            img_rgb: RGB numpy array (H, W, 3), uint8

        Returns:
            (x1, y1, x2, y2) в пикселях или None
        """
        try:
            results = self._model(
                img_rgb,
                verbose=False,
                conf=config.YOLO_CONF_THRESHOLD,
                iou=config.YOLO_IOU_THRESHOLD,
            )

            if not results or len(results) == 0:
                logger.info("YOLO: объекты не найдены")
                return None

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                logger.info("YOLO: boxes пусты")
                return None

            # Собираем все детекции
            all_boxes = self._parse_boxes(boxes)
            if not all_boxes:
                return None

            img_h, img_w = img_rgb.shape[:2]

            # Шаг 1: ищем среди предпочтительных классов
            preferred = [
                b for b in all_boxes
                if b['cls'] in config.YOLO_PREFERRED_CLASSES
            ]

            candidates = preferred if preferred else all_boxes

            # Шаг 2: выбираем по стратегии
            if config.YOLO_SELECTION_STRATEGY == "center":
                best = self._select_by_center(candidates, img_w, img_h)
            else:
                best = self._select_by_confidence(candidates)

            bbox = best['bbox']
            logger.info(
                f"YOLO: bbox={bbox}, cls={best['cls']}, "
                f"conf={best['conf']:.3f}, "
                f"strategy={config.YOLO_SELECTION_STRATEGY}, "
                f"preferred={'yes' if preferred else 'no'}"
            )
            return bbox

        except Exception as e:
            logger.error(f"Ошибка детекции YOLO: {e}")
            return None

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _parse_boxes(self, boxes) -> List[dict]:
        """Преобразовать boxes из ultralytics в список словарей."""
        result = []
        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy_all)):
            x1, y1, x2, y2 = (
                int(xyxy_all[i][0]), int(xyxy_all[i][1]),
                int(xyxy_all[i][2]), int(xyxy_all[i][3])
            )
            result.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(conf_all[i]),
                'cls': int(cls_all[i]),
            })
        return result

    def _select_by_center(
        self,
        candidates: List[dict],
        img_w: int,
        img_h: int,
    ) -> dict:
        """
        Выбрать bbox, центр которого ближе всего к центру изображения.

        Пользователь обычно центрирует номер дома в кадре.
        """
        cx_img = img_w / 2.0
        cy_img = img_h / 2.0

        def dist_to_center(b: dict) -> float:
            x1, y1, x2, y2 = b['bbox']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return (cx - cx_img) ** 2 + (cy - cy_img) ** 2

        return min(candidates, key=dist_to_center)

    def _select_by_confidence(self, candidates: List[dict]) -> dict:
        """Выбрать bbox с наибольшей уверенностью."""
        return max(candidates, key=lambda b: b['conf'])
