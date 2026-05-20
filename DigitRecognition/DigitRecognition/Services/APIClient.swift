//
//  APIClient.swift
//  DigitRecognition
//
//  Клиент для взаимодействия с сервером распознавания
//

import Foundation
import UIKit

class APIClient {
    
    static let shared = APIClient()
    
    private var baseURL: String
    private let session: URLSession
    
    private init() {
        // Замените на IP-адрес вашего ноутбука
        self.baseURL = "http://YOUR_SERVER_IP:5000"
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: configuration)
    }
    
    func updateServerURL(_ url: String) {
        self.baseURL = url
    }
    
    func checkHealth(completion: @escaping (Result<HealthResponse, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/health") else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        session.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                completion(.failure(APIError.serverError))
                return
            }
            
            guard let data = data else {
                completion(.failure(APIError.noData))
                return
            }
            
            do {
                let healthResponse = try JSONDecoder().decode(HealthResponse.self, from: data)
                completion(.success(healthResponse))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
    
    func predictDigit(image: UIImage, completion: @escaping (Result<PredictionResponse, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/predict") else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        // Конвертируем изображение в PNG данные
        guard let imageData = image.pngData() else {
            completion(.failure(APIError.imageConversionFailed))
            return
        }
        
        // Создаем multipart/form-data запрос
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Добавляем изображение
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Завершаем тело запроса
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        session.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(APIError.serverError))
                return
            }
            
            guard let data = data else {
                completion(.failure(APIError.noData))
                return
            }
            
            // Обрабатываем ответ
            if (200...299).contains(httpResponse.statusCode) {
                do {
                    let predictionResponse = try JSONDecoder().decode(PredictionResponse.self, from: data)
                    if predictionResponse.success {
                        completion(.success(predictionResponse))
                    } else {
                        completion(.failure(APIError.predictionFailed))
                    }
                } catch {
                    completion(.failure(error))
                }
            } else {
                // Пытаемся декодировать ошибку
                if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                    completion(.failure(APIError.serverMessage(errorResponse.error)))
                } else {
                    completion(.failure(APIError.serverError))
                }
            }
        }.resume()
    }
}

// MARK: - Supporting Types

struct HealthResponse: Codable {
    let status: String
    let model: ModelInfo
}

struct ModelInfo: Codable {
    let inputShape: [String?]
    let outputShape: [String?]
    let numParams: Int
    let modelPath: String
    
    enum CodingKeys: String, CodingKey {
        case inputShape = "input_shape"
        case outputShape = "output_shape"
        case numParams = "num_params"
        case modelPath = "model_path"
    }
}

enum APIError: Error, LocalizedError {
    case invalidURL
    case noData
    case imageConversionFailed
    case serverError
    case predictionFailed
    case serverMessage(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Некорректный URL сервера"
        case .noData:
            return "Нет данных от сервера"
        case .imageConversionFailed:
            return "Не удалось конвертировать изображение"
        case .serverError:
            return "Ошибка сервера"
        case .predictionFailed:
            return "Не удалось выполнить распознавание"
        case .serverMessage(let message):
            return message
        }
    }
}
