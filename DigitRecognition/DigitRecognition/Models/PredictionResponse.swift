//
//  PredictionResponse.swift
//  DigitRecognition
//
//  Модель ответа от сервера распознавания цифр
//

import Foundation

struct PredictionResponse: Codable {
    let success: Bool
    let predictions: [Double]
    let predictedClass: Int
    let confidence: Double
    
    enum CodingKeys: String, CodingKey {
        case success
        case predictions
        case predictedClass = "predicted_class"
        case confidence
    }
}

struct ErrorResponse: Codable {
    let success: Bool
    let error: String
}
