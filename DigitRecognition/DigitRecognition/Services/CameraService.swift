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
