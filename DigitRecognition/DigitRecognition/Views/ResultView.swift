//
//  ResultView.swift
//  DigitRecognition
//
//  View для отображения результатов распознавания
//

import SwiftUI

struct ResultView: View {
    let prediction: PredictionResponse
    let onDismiss: () -> Void
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Предсказанная цифра
                VStack(spacing: 10) {
                    Text("Распознанная цифра")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text("\(prediction.predictedClass)")
                        .font(.system(size: 72, weight: .bold, design: .rounded))
                        .foregroundColor(.primary)
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(16)
                .shadow(radius: 5)
                
                // Уверенность
                VStack(spacing: 8) {
                    Text("Уверенность")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Text("\(Int(prediction.confidence * 100))%")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(prediction.confidence > 0.8 ? .green : .orange)
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(12)
                
                // График вероятностей
                VStack(alignment: .leading, spacing: 12) {
                    Text("Распределение вероятностей")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    ForEach(0..<10) { digit in
                        ProbabilityBar(
                            digit: digit,
                            probability: prediction.predictions[digit],
                            isPredicted: digit == prediction.predictedClass
                        )
                    }
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(16)
                .shadow(radius: 3)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Результаты")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Закрыть") {
                        onDismiss()
                    }
                }
            }
        }
    }
}

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
                        .frame(width: geometry.size.width * CGFloat(probability), height: 8)
                        .cornerRadius(4)
                        .animation(.easeInOut(duration: 0.3), value: probability)
                }
            }
            
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
                predictions: [0.01, 0.02, 0.85, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
                predictedClass: 2,
                confidence: 0.85
            ),
            onDismiss: {}
        )
    }
}
