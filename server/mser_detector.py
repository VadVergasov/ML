"""
Детектор текстовых регионов на базе OpenCV MSER.

MSER (Maximally Stable Extremal Regions) специально разработан
для поиска текстоподобных областей на изображениях.

Алгоритм v2:
  1. Предобработка: grayscale + CLAHE + bilateral filter
  2. MSER на оригинале и инвертированном изображении (тёмный/светлый текст)
  3. Фильтрация по размеру, соотношению сторон, заполненности
  4. Группировка по вертикальному перекрытию (цифры одного числа на одной строке)
  5. Выбор лучшей группы по количеству регионов и компактности
  6. Расширение bbox с отступом
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Тип bbox: (x1, y1, x2, y2) в пикселях
BBox = Tuple[int, int, int, int]


class MSERDetector:
    """Детектор текстовых регионов через OpenCV MSER."""

    def __init__(self):
        # В OpenCV 4.x параметры передаются позиционно:
        # MSER_create(delta, min_area, max_area)
        # delta=5 — более чувствительный (меньше = больше регионов)
        self._mser = cv2.MSER_create(
            config.MSER_DELTA,
            config.MSER_MIN_AREA,
            config.MSER_MAX_AREA,
        )

    def detect(self, img_rgb: np.ndarray) -> Optional[BBox]:
        """
        Найти bbox области с цифрами/текстом на изображении.

        Args:
            img_rgb: RGB numpy array (H, W, 3), uint8

        Returns:
            (x1, y1, x2, y2) в пикселях или None если текст не найден
        """
        try:
            h, w = img_rgb.shape[:2]

            # --- Предобработка ---
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Bilateral filter — сглаживает шум, сохраняет края
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # CLAHE — улучшает контраст локально
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)

            # --- MSER на прямом и инвертированном изображении ---
            # Прямой: тёмный текст на светлом фоне
            regions_dark, _ = self._mser.detectRegions(gray_eq)
            # Инвертированный: светлый текст на тёмном фоне
            regions_light, _ = self._mser.detectRegions(
                cv2.bitwise_not(gray_eq)
            )

            all_regions = list(regions_dark) + list(regions_light)

            if not all_regions:
                logger.info("MSER: регионы не найдены")
                return None

            # --- Получаем bounding rect для каждого региона ---
            rects = []
            for region in all_regions:
                pts = region.reshape(-1, 1, 2)
                x, y, rw, rh = cv2.boundingRect(pts)
                rects.append((x, y, rw, rh))

            # --- Фильтрация ---
            filtered = self._filter_rects(rects, w, h)
            logger.info(
                f"MSER: всего регионов={len(rects)}, "
                f"после фильтрации={len(filtered)}"
            )

            if not filtered:
                logger.info("MSER: нет регионов после фильтрации")
                return None

            # --- Группировка по горизонтальным строкам ---
            groups = self._group_by_row(filtered, h)
            logger.info(f"MSER: групп по строкам={len(groups)}")

            if not groups:
                return None

            # --- Выбор лучшей группы ---
            best_bbox = self._select_best_group(groups, w, h)

            if best_bbox is None:
                return None

            logger.info(f"MSER: итоговый bbox={best_bbox}")
            return best_bbox

        except Exception as e:
            logger.error(f"Ошибка детекции MSER: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _filter_rects(
        self,
        rects: List[Tuple[int, int, int, int]],
        img_w: int,
        img_h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Отфильтровать прямоугольники по размеру и соотношению сторон.

        Цифры имеют характерные пропорции:
        - высота: 3%–70% высоты изображения
        - ширина: 1%–50% ширины изображения
        - соотношение ширина/высота: 0.1–3.0 (цифры примерно квадратные)
        """
        result = []
        min_h = img_h * config.MSER_MIN_HEIGHT_RATIO
        max_h = img_h * config.MSER_MAX_HEIGHT_RATIO
        min_w = img_w * config.MSER_MIN_WIDTH_RATIO
        max_w = img_w * 0.5  # не более половины ширины изображения

        for x, y, rw, rh in rects:
            # Фильтр по высоте
            if not (min_h <= rh <= max_h):
                continue
            # Фильтр по ширине
            if not (min_w <= rw <= max_w):
                continue
            # Фильтр по соотношению сторон
            aspect = rw / max(rh, 1)
            if not (0.1 <= aspect <= config.MSER_MAX_ASPECT_RATIO):
                continue
            result.append((x, y, rw, rh))

        return result

    def _group_by_row(
        self,
        rects: List[Tuple[int, int, int, int]],
        img_h: int,
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Сгруппировать прямоугольники по горизонтальным строкам.

        Два прямоугольника попадают в одну строку если их вертикальные
        диапазоны перекрываются более чем на 50%.

        Returns:
            Список групп, каждая группа — список (x, y, w, h)
        """
        if not rects:
            return []

        # Сортируем по y (верхний край)
        sorted_rects = sorted(rects, key=lambda r: r[1])

        groups: List[List[Tuple[int, int, int, int]]] = []

        for rect in sorted_rects:
            x, y, rw, rh = rect
            y2 = y + rh
            placed = False

            for group in groups:
                # Берём медианный вертикальный диапазон группы
                gy1 = min(r[1] for r in group)
                gy2 = max(r[1] + r[3] for r in group)
                gh = gy2 - gy1

                # Перекрытие по вертикали
                overlap_y1 = max(y, gy1)
                overlap_y2 = min(y2, gy2)
                overlap = max(0, overlap_y2 - overlap_y1)

                # Порог перекрытия: 40% от меньшего из двух высот
                min_h = min(rh, gh)
                if overlap >= 0.4 * min_h:
                    group.append(rect)
                    placed = True
                    break

            if not placed:
                groups.append([rect])

        # Оставляем только группы с минимальным количеством регионов
        min_count = config.MSER_MIN_GROUP_SIZE
        groups = [g for g in groups if len(g) >= min_count]

        return groups

    def _select_best_group(
        self,
        groups: List[List[Tuple[int, int, int, int]]],
        img_w: int,
        img_h: int,
    ) -> Optional[BBox]:
        """
        Выбрать лучшую группу и вернуть её bbox.

        Критерии выбора (в порядке приоритета):
        1. Группа не превышает максимальный размер bbox
        2. Наибольшее количество регионов (больше цифр = лучше)
        3. Наибольшая плотность (count / area)
        """
        max_bbox_w = img_w * config.MSER_MAX_BBOX_WIDTH_RATIO
        max_bbox_h = img_h * config.MSER_MAX_BBOX_HEIGHT_RATIO

        candidates = []
        for group in groups:
            x1 = min(r[0] for r in group)
            y1 = min(r[1] for r in group)
            x2 = max(r[0] + r[2] for r in group)
            y2 = max(r[1] + r[3] for r in group)
            bbox = (x1, y1, x2, y2)
            bw = x2 - x1
            bh = y2 - y1
            area = max(bw * bh, 1)
            count = len(group)
            density = count / area

            candidates.append({
                'bbox': bbox,
                'count': count,
                'area': area,
                'density': density,
                'bw': bw,
                'bh': bh,
            })

        # Фильтруем по максимальному размеру
        valid = [
            c for c in candidates
            if c['bw'] <= max_bbox_w and c['bh'] <= max_bbox_h
        ]

        if not valid:
            # Все слишком большие — берём наименьший по площади
            logger.info(
                "MSER: все группы превышают макс. размер, "
                "берём наименьшую по площади"
            )
            valid = candidates
            best = min(valid, key=lambda c: c['area'])
        else:
            # Сортируем: сначала по количеству (убывание),
            # при равенстве — по плотности (убывание)
            valid.sort(key=lambda c: (c['count'], c['density']), reverse=True)
            best = valid[0]

        logger.info(
            f"MSER: выбрана группа count={best['count']}, "
            f"density={best['density']:.6f}, bbox={best['bbox']}"
        )
        return best['bbox']
