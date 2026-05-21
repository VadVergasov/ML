//
//  PredictionResponse.swift
//  DigitRecognition
//
//  Модели ответов от сервера распознавания цифр
//

import Foundation

// MARK: - Bounding Box от YOLOv8

struct BoundingBox: Codable {
    let x1: Int
    let y1: Int
    let x2: Int
    let y2: Int

    /// Ширина bbox в пикселях
    var width: Int { x2 - x1 }
    /// Высота bbox в пикселях
    var height: Int { y2 - y1 }
}

// MARK: - Результат распознавания одной цифры

struct DigitResult: Codable {
    let digit: Int
    let confidence: Double
    /// 11 вероятностей: индексы 0-9 — цифры, индекс 10 — заглушка
    let probabilities: [Double]
}

// MARK: - Ответ сервера на запрос распознавания

struct PredictionResponse: Codable {
    let success: Bool
    let digits: [DigitResult]
    let number: String
    let digitsCount: Int
    /// Bounding box найденной области с цифрами (nil если YOLO не нашёл объект)
    let bbox: BoundingBox?

    enum CodingKeys: String, CodingKey {
        case success
        case digits
        case number
        case digitsCount = "digits_count"
        case bbox
    }
}

// MARK: - Ответ об ошибке

struct ErrorResponse: Codable {
    let success: Bool
    let error: String
}
