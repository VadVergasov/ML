//
//  MainView.swift
//  DigitRecognition
//
//  Главный View приложения
//

import SwiftUI
import PhotosUI

struct MainView: View {
    @State private var selectedImage: UIImage?
    /// Нормализованное изображение (EXIF применён) — именно его видит сервер.
    /// Используется для отображения bbox в ResultView.
    @State private var normalizedImage: UIImage?
    @State private var prediction: PredictionResponse?
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showResult = false
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var serverURL = "http://YOUR_SERVER_IP:8888"

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    // MARK: - Настройки сервера (вверху, всегда видно)
                    VStack(alignment: .leading, spacing: 6) {
                        Label("Адрес сервера", systemImage: "network")
                            .font(.headline)
                            .foregroundColor(.primary)

                        TextField("http://192.168.1.100:5000", text: $serverURL)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.URL)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .font(.system(.body, design: .monospaced))
                            .onChange(of: serverURL) { newValue in
                                APIClient.shared.updateServerURL(newValue)
                            }

                        Text("Введите IP-адрес ноутбука в локальной сети")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)
                    .padding(.top, 8)

                    // MARK: - Область изображения
                    ZStack {
                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 280)
                                .cornerRadius(12)
                                .shadow(radius: 4)
                        } else {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(.systemGray6))
                                .frame(height: 280)
                                .overlay(
                                    VStack(spacing: 12) {
                                        Image(systemName: "photo.badge.plus")
                                            .font(.system(size: 48))
                                            .foregroundColor(.gray)
                                        Text("Нет изображения")
                                            .font(.headline)
                                            .foregroundColor(.secondary)
                                        Text("Выберите из галереи или сфотографируйте")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                            .multilineTextAlignment(.center)
                                    }
                                )
                        }
                    }
                    .padding(.horizontal)

                    // MARK: - Кнопки выбора изображения
                    HStack(spacing: 12) {
                        Button(action: { showImagePicker = true }) {
                            Label("Галерея", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isLoading)

                        Button(action: { showCamera = true }) {
                            Label("Камера", systemImage: "camera")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                    }
                    .padding(.horizontal)

                    // MARK: - Кнопка распознавания
                    Button(action: recognizeDigit) {
                        Group {
                            if isLoading {
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .progressViewStyle(
                                            CircularProgressViewStyle(tint: .white)
                                        )
                                    Text("Распознавание...")
                                }
                            } else {
                                Label("Распознать", systemImage: "wand.and.stars")
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(selectedImage == nil || isLoading)
                    .padding(.horizontal)

                    // MARK: - Сообщение об ошибке
                    if let error = errorMessage {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red)
                                .multilineTextAlignment(.leading)
                        }
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    }

                    Spacer(minLength: 20)
                }
            }
            .navigationTitle("Распознавание цифр")
            .navigationBarTitleDisplayMode(.inline)
            // Выбор из галереи
            .sheet(isPresented: $showImagePicker) {
                PHPickerRepresentable { image in
                    selectedImage = image
                    errorMessage = nil
                }
            }
            // Съёмка камерой
            .sheet(isPresented: $showCamera) {
                CameraPickerRepresentable { image in
                    selectedImage = image
                    errorMessage = nil
                }
            }
            // Результат распознавания
            .sheet(isPresented: $showResult) {
                if let prediction = prediction {
                    ResultView(
                        prediction: prediction,
                        // Передаём нормализованное изображение —
                        // bbox рассчитан именно для него
                        originalImage: normalizedImage ?? selectedImage
                    ) {
                        showResult = false
                    }
                }
            }
        }
    }

    private func recognizeDigit() {
        guard let image = selectedImage else { return }

        isLoading = true
        errorMessage = nil

        // Нормализуем ориентацию и ресайзим — точно так же как APIClient.
        // normalizedImage используется в ResultView для отображения bbox,
        // поэтому он должен совпадать с изображением, которое видит сервер.
        let normalized = image.normalizedOrientation()
            .resizedToMaxSide(1920)
        normalizedImage = normalized

        APIClient.shared.updateServerURL(serverURL)

        APIClient.shared.predictDigit(image: image) { result in
            DispatchQueue.main.async {
                isLoading = false
                switch result {
                case .success(let response):
                    self.prediction = response
                    self.showResult = true
                case .failure(let error):
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - PHPickerRepresentable (галерея)

struct PHPickerRepresentable: UIViewControllerRepresentable {
    let onImagePicked: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration()
        configuration.filter = .images
        configuration.selectionLimit = 1
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: PHPickerViewController,
        context: Context
    ) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onImagePicked: onImagePicked)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let onImagePicked: (UIImage?) -> Void

        init(onImagePicked: @escaping (UIImage?) -> Void) {
            self.onImagePicked = onImagePicked
        }

        func picker(
            _ picker: PHPickerViewController,
            didFinishPicking results: [PHPickerResult]
        ) {
            picker.dismiss(animated: true)
            guard let result = results.first else {
                onImagePicked(nil)
                return
            }
            guard result.itemProvider.canLoadObject(ofClass: UIImage.self) else {
                onImagePicked(nil)
                return
            }
            result.itemProvider.loadObject(ofClass: UIImage.self) { object, _ in
                DispatchQueue.main.async {
                    self.onImagePicked(object as? UIImage)
                }
            }
        }
    }
}

// MARK: - CameraPickerRepresentable (камера)

struct CameraPickerRepresentable: UIViewControllerRepresentable {
    let onImagePicked: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context
    ) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onImagePicked: onImagePicked)
    }

    class Coordinator: NSObject,
        UIImagePickerControllerDelegate,
        UINavigationControllerDelegate
    {
        let onImagePicked: (UIImage?) -> Void

        init(onImagePicked: @escaping (UIImage?) -> Void) {
            self.onImagePicked = onImagePicked
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            picker.dismiss(animated: true)
            onImagePicked(info[.originalImage] as? UIImage)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
            onImagePicked(nil)
        }
    }
}

// MARK: - Preview

struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
