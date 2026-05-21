//
//  CameraService.swift
//  DigitRecognition
//
//  Сервис для работы с камерой устройства
//

import UIKit
import AVFoundation
import PhotosUI

class CameraService: NSObject {
    
    static let shared = CameraService()
    
    private var imagePickerController: UIImagePickerController?
    private var completion: ((UIImage?) -> Void)?
    
    private override init() {
        super.init()
    }
    
    // MARK: - Photo Library
    
    func pickImage(from viewController: UIViewController, completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
        
        var configuration = PHPickerConfiguration()
        configuration.filter = .images
        configuration.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = self
        viewController.present(picker, animated: true)
    }
    
    // MARK: - Camera
    
    func captureImage(from viewController: UIViewController, completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
        
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            completion(nil)
            return
        }
        
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        viewController.present(picker, animated: true)
    }
    
    // MARK: - Check Permissions
    
    func checkCameraPermission(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        switch status {
        case .authorized:
            completion(true)
        case .denied, .restricted:
            completion(false)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        @unknown default:
            completion(false)
        }
    }
    
    func checkPhotoLibraryPermission(completion: @escaping (Bool) -> Void) {
        let status = PHPhotoLibrary.authorizationStatus()
        
        switch status {
        case .authorized, .limited:
            completion(true)
        case .denied, .restricted:
            completion(false)
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization { status in
                DispatchQueue.main.async {
                    completion(status == .authorized || status == .limited)
                }
            }
        @unknown default:
            completion(false)
        }
    }
}

// MARK: - PHPickerViewControllerDelegate

extension CameraService: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        
        guard let result = results.first else {
            completion?(nil)
            return
        }
        
        if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
            result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self?.completion?(nil)
                    }
                    return
                }
                
                DispatchQueue.main.async {
                    self?.completion?(object as? UIImage)
                }
            }
        } else {
            completion?(nil)
        }
    }
}

// MARK: - UIImagePickerControllerDelegate

extension CameraService: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        
        if let image = info[.originalImage] as? UIImage {
            completion?(image)
        } else {
            completion?(nil)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
        completion?(nil)
    }
}

// MARK: - UIImage orientation normalization & resize

extension UIImage {
    /// Вернуть копию изображения с применённой EXIF-ориентацией.
    ///
    /// JPEG-фото с камеры хранят пиксели в "сыром" виде и указывают
    /// нужный поворот через тег Orientation. UIKit применяет его
    /// автоматически при отображении, но pngData()/jpegData() отдают
    /// пиксели без поворота. Это приводит к тому, что сервер получает
    /// неповёрнутое изображение, а bbox рисуется поверх повёрнутого.
    ///
    /// Метод перерисовывает изображение в UIGraphicsImageRenderer,
    /// применяя трансформацию ориентации, и возвращает .up-изображение.
    func normalizedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }

        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }

    /// Уменьшить изображение так, чтобы длинная сторона не превышала
    /// `maxSide` пикселей. Если изображение уже меньше — возвращает self.
    ///
    /// Используется перед отправкой на сервер: PNG с iPhone 12MP весит
    /// ~36MB и вызывает 413 Request Entity Too Large. После ресайза до
    /// 1920px PNG весит ~3-6MB. Bbox рассчитывается для ресайзнутого
    /// изображения, поэтому координаты остаются точными.
    func resizedToMaxSide(_ maxSide: CGFloat) -> UIImage {
        let longSide = max(size.width, size.height)
        guard longSide > maxSide else { return self }

        let scale = maxSide / longSide
        let newSize = CGSize(
            width: (size.width * scale).rounded(),
            height: (size.height * scale).rounded()
        )
        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
    }
}
