//
//  ResultView.swift
//  DigitRecognition
//
//  View для отображения результатов распознавания номера дома
//

import SwiftUI

struct ResultView: View {
    let prediction: PredictionResponse
    let originalImage: UIImage?
    let onDismiss: () -> Void

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    // MARK: - Изображение с выделенным bbox (если есть)
                    if let image = originalImage {
                        ImageWithBBox(
                            image: image,
                            bbox: prediction.bbox
                        )
                        // Фиксированная высота нужна чтобы GeometryReader
                        // внутри ImageWithBBox получил корректный containerH
                        // при первом рендере (maxHeight даёт 0 в VStack).
                        .frame(height: 260)
                        .padding(.horizontal)
                        .shadow(radius: 4)
                    }

                    // MARK: - Распознанный номер дома
                    VStack(spacing: 8) {
                        Text("Номер дома")
                            .font(.headline)
                            .foregroundColor(.secondary)

                        Text(prediction.number.isEmpty ? "—" : prediction.number)
                            .font(.system(size: 72, weight: .bold, design: .rounded))
                            .foregroundColor(.primary)
                            .minimumScaleFactor(0.4)
                            .lineLimit(1)
                            .padding(.horizontal)

                        Text("\(prediction.digitsCount) цифр(ы) найдено")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if prediction.bbox == nil {
                            Label(
                                "Область не выделена (MSER не нашёл текст)",
                                systemImage: "exclamationmark.triangle"
                            )
                            .font(.caption2)
                            .foregroundColor(.orange)
                        }
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemBackground))
                    .cornerRadius(16)
                    .shadow(radius: 4)
                    .padding(.horizontal)

                    // MARK: - Детали по каждой цифре
                    if prediction.digits.count > 1 {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Детали распознавания")
                                .font(.headline)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)

                            ForEach(
                                Array(prediction.digits.enumerated()),
                                id: \.offset
                            ) { index, digitResult in
                                DigitDetailCard(
                                    position: index + 1,
                                    result: digitResult
                                )
                                .padding(.horizontal)
                            }
                        }
                    } else if let first = prediction.digits.first {
                        // Одна цифра — показываем вероятности (только 0-9)
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Распределение вероятностей")
                                .font(.headline)
                                .foregroundColor(.secondary)

                            // probabilities содержит 11 элементов (0-9 + заглушка),
                            // показываем только цифры 0-9 (индексы 0..<10)
                            ForEach(0..<10) { digit in
                                ProbabilityBar(
                                    digit: digit,
                                    probability: first.probabilities.count > digit
                                        ? first.probabilities[digit]
                                        : 0.0,
                                    isPredicted: digit == first.digit
                                )
                            }
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .cornerRadius(16)
                        .shadow(radius: 3)
                        .padding(.horizontal)
                    }

                    Spacer(minLength: 20)
                }
                .padding(.top)
            }
            .navigationTitle("Результат")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Закрыть") { onDismiss() }
                }
            }
        }
    }
}

// MARK: - Изображение с наложенным bbox

/// Отображает изображение с зелёной рамкой bbox поверх него.
///
/// Ключевая сложность: SwiftUI Image с .aspectRatio(.fit) занимает
/// меньше места чем контейнер. GeometryReader даёт размер контейнера,
/// а не изображения. Поэтому вычисляем реальный размер изображения
/// вручную и передаём его в BBoxOverlay.
struct ImageWithBBox: View {
    let image: UIImage
    let bbox: BoundingBox?

    // Размер изображения в пикселях (то, что получил сервер)
    private var pixelSize: CGSize {
        CGSize(
            width: CGFloat(image.cgImage?.width
                ?? Int(image.size.width * image.scale)),
            height: CGFloat(image.cgImage?.height
                ?? Int(image.size.height * image.scale))
        )
    }

    var body: some View {
        GeometryReader { geo in
            let containerW = geo.size.width
            let containerH = geo.size.height
            let px = pixelSize.width
            let py = pixelSize.height

            // Вычисляем реальный размер изображения при aspectFit
            let scaleToFit = min(
                containerW / max(px, 1),
                containerH / max(py, 1)
            )
            let imgW = px * scaleToFit
            let imgH = py * scaleToFit

            // Смещение для центрирования изображения в контейнере
            let offsetX = (containerW - imgW) / 2
            let offsetY = (containerH - imgH) / 2

            ZStack(alignment: .topLeading) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .cornerRadius(12)
                    .frame(width: containerW, height: containerH)

                if let bbox = bbox {
                    // BBoxOverlay рисует рамку в координатах пикселей изображения.
                    // Смещаем его на offsetX/offsetY чтобы совпало с изображением.
                    BBoxOverlay(
                        bbox: bbox,
                        imagePixelSize: pixelSize,
                        displaySize: CGSize(width: imgW, height: imgH)
                    )
                    .offset(x: offsetX, y: offsetY)
                }
            }
        }
    }
}

// MARK: - Оверлей bounding box поверх изображения

struct BBoxOverlay: View {
    let bbox: BoundingBox
    /// Размер изображения в пикселях (то, что получил сервер через pngData)
    let imagePixelSize: CGSize
    /// Реальный отображаемый размер Image-вью в points (уже вычислен снаружи)
    let displaySize: CGSize

    var body: some View {
        // Масштаб: пиксели → points
        // displaySize уже учитывает соотношение сторон (aspectFit),
        // поэтому scaleX == scaleY, но считаем оба для надёжности.
        let scaleX = displaySize.width / max(imagePixelSize.width, 1)
        let scaleY = displaySize.height / max(imagePixelSize.height, 1)

        // Координаты рамки в points (левый верхний угол + размер)
        let rx = CGFloat(bbox.x1) * scaleX
        let ry = CGFloat(bbox.y1) * scaleY
        let rw = CGFloat(bbox.width) * scaleX
        let rh = CGFloat(bbox.height) * scaleY

        // Canvas размером с изображение — рисуем рамку в абсолютных координатах
        Canvas { context, _ in
            let rect = CGRect(x: rx, y: ry, width: rw, height: rh)
            var path = Path()
            path.addRect(rect)
            context.stroke(
                path,
                with: .color(.green),
                lineWidth: 2
            )
        }
        .frame(width: displaySize.width, height: displaySize.height)
    }
}

// MARK: - Карточка одной цифры

struct DigitDetailCard: View {
    let position: Int
    let result: DigitResult

    var body: some View {
        HStack(spacing: 16) {
            // Позиция
            Text("#\(position)")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 28)

            // Цифра
            Text("\(result.digit)")
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .frame(width: 44)

            // Уверенность
            VStack(alignment: .leading, spacing: 4) {
                Text("Уверенность")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(.systemGray5))
                            .frame(height: 8)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(
                                result.confidence > 0.8
                                    ? Color.green
                                    : result.confidence > 0.5
                                        ? Color.orange
                                        : Color.red
                            )
                            .frame(
                                width: geo.size.width * CGFloat(result.confidence),
                                height: 8
                            )
                    }
                }
                .frame(height: 8)
            }

            // Процент
            Text("\(Int(result.confidence * 100))%")
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
                .foregroundColor(
                    result.confidence > 0.8 ? .green
                    : result.confidence > 0.5 ? .orange
                    : .red
                )
                .frame(width: 44, alignment: .trailing)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Полоска вероятности

struct ProbabilityBar: View {
    let digit: Int
    let probability: Double
    let isPredicted: Bool

    var body: some View {
        HStack(spacing: 12) {
            Text("\(digit)")
                .font(.system(.body, design: .monospaced))
                .frame(width: 20)
                .foregroundColor(isPredicted ? .primary : .secondary)

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 8)
                        .cornerRadius(4)
                    Rectangle()
                        .fill(isPredicted ? Color.blue : Color(.systemGray4))
                        .frame(
                            width: geometry.size.width * CGFloat(probability),
                            height: 8
                        )
                        .cornerRadius(4)
                        .animation(.easeInOut(duration: 0.3), value: probability)
                }
            }
            .frame(height: 8)

            Text("\(Int(probability * 100))%")
                .font(.system(.caption, design: .monospaced))
                .frame(width: 40, alignment: .trailing)
                .foregroundColor(isPredicted ? .primary : .secondary)
        }
    }
}

// MARK: - Preview

struct ResultView_Previews: PreviewProvider {
    static var previews: some View {
        ResultView(
            prediction: PredictionResponse(
                success: true,
                digits: [
                    DigitResult(
                        digit: 4,
                        confidence: 0.97,
                        // 11 вероятностей: индексы 0-9 + заглушка (индекс 10)
                        probabilities: [
                            0.0, 0.0, 0.0, 0.0, 0.97, 0.01, 0.01, 0.0, 0.0, 0.01,
                            0.0
                        ]
                    ),
                    DigitResult(
                        digit: 2,
                        confidence: 0.85,
                        probabilities: [
                            0.01, 0.02, 0.85, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01,
                            0.01, 0.0
                        ]
                    )
                ],
                number: "42",
                digitsCount: 2,
                bbox: BoundingBox(x1: 10, y1: 20, x2: 150, y2: 180)
            ),
            originalImage: nil,
            onDismiss: {}
        )
    }
}
