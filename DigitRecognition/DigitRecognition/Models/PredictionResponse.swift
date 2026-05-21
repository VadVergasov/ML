//
//  PredictionResponse.swift
//  DigitRecognition
//
//  Модели ответов от сервера распознавания цифр
//

import Foundation

// MARK: - Результат распознавания одной цифры

struct DigitResult: Codable {
    let digit: Int
    let confidence: Double
    let probabilities: [Double]
}

// MARK: - Ответ сервера на запрос распознавания

struct PredictionResponse: Codable {
    let success: Bool
    let digits: [DigitResult]
    let number: String
    let digitsCount: Int

    enum CodingKeys: String, CodingKey {
        case success
        case digits
        case number
        case digitsCount = "digits_count"
    }
}

// MARK: - Ответ об ошибке

struct ErrorResponse: Codable {
    let success: Bool
    let error: String
}
