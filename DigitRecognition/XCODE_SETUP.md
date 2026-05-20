# Инструкция по созданию Xcode проекта DigitRecognition

Эта инструкция поможет вам создать Xcode проект и добавить существующие Swift файлы.

## Шаг 1: Создание нового проекта в Xcode

1. Откройте Xcode
2. Выберите **File → New → Project...** (или нажмите `⌘⇧N`)
3. В разделе **iOS** выберите **App**
4. Нажмите **Next**

### Настройки проекта:

| Поле | Значение |
|------|----------|
| Product Name | `DigitRecognition` |
| Team | Ваша команда разработчика |
| Organization Identifier | `com.yourcompany` (или любой другой) |
| Bundle Identifier | `com.yourcompany.DigitRecognition` |
| Interface | `SwiftUI` |
| Language | `Swift` |
| Storage | `None` |
| Include Tests | ✅ (по желанию) |

5. Нажмите **Next**
6. Выберите папку для сохранения проекта (рекомендуется `/Users/vadvergasov/workspace/ML/DigitRecognition`)
7. Убедитесь, что галочка **Create Git repository** снята (так как проект уже в Git)
8. Нажмите **Create**

## Шаг 2: Удаление автоматически созданных файлов

Xcode создаст несколько файлов по умолчанию. Их нужно удалить:

1. В навигаторе проекта (слева) найдите файл `DigitRecognitionApp.swift` (или `ContentView.swift`)
2. Удалите его (правый клик → Delete → Move to Trash)
3. Удалите файл `Assets.xcassets` (если он не нужен)
4. Удалите файл `Preview Content` (если он есть)

## Шаг 3: Добавление существующих Swift файлов

### Структура папок проекта должна быть такой:

```
DigitRecognition/
├── DigitRecognition/
│   ├── Models/
│   │   └── PredictionResponse.swift
│   ├── Services/
│   │   ├── APIClient.swift
│   │   └── CameraService.swift
│   ├── Views/
│   │   ├── MainView.swift
│   │   └── ResultView.swift
│   └── DigitRecognitionApp.swift
└── Info.plist
```

### Добавление файлов:

1. В Finder откройте папку `/Users/vadvergasov/workspace/ML/DigitRecognition/DigitRecognition/`
2. В Xcode навигаторе проекта найдите папку `DigitRecognition` (синяя иконка)
3. Перетащите следующие папки из Finder в Xcode:
   - `Models/`
   - `Services/`
   - `Views/`
   - `DigitRecognitionApp.swift`

4. В появившемся диалоге выберите:
   - ✅ **Copy items if needed** — **НЕ** отмечать (файлы уже на месте)
   - ✅ **Create groups** (не Create folder references)
   - ✅ **Add to targets: DigitRecognition**

5. Нажмите **Finish**

## Шаг 4: Настройка Info.plist

1. В навигаторе проекта найдите файл `Info.plist` (в папке проекта)
2. Если его нет, создайте:
   - Правый клик на папку проекта → New File
   - Выберите **Property List**
   - Назовите `Info.plist`
   - Скопируйте содержимое из файла `/Users/vadvergasov/workspace/ML/DigitRecognition/Info.plist`

3. Убедитесь, что в Info.plist есть следующие ключи:
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>Приложению нужен доступ к камере для фотографирования цифр для распознавания</string>
   
   <key>NSPhotoLibraryUsageDescription</key>
   <string>Приложению нужен доступ к фотогалерее для выбора изображений цифр для распознавания</string>
   ```

4. В настройках проекта (выберите синий иконку проекта вверху):
   - Перейдите на вкладку **Build Settings**
   - Найдите **Info.plist File**
   - Установите значение: `DigitRecognition/Info.plist`

## Шаг 5: Проверка настроек проекта

1. Выберите синий иконку проекта в навигаторе
2. Перейдите на вкладку **General**
3. Проверьте:
   - **Deployment Info** → **Minimum Deployments**: iOS 15.0 или выше
   - **Deployment Info** → **Devices**: iPhone (или Universal)
   - **App Icons and Launch Images** → **Launch Screen**: настроен

## Шаг 6: Сборка и запуск

1. Выберите симулятор (например, iPhone 15 Pro)
2. Нажмите `⌘R` или кнопку **Run** (▶️)
3. Приложение должно запуститься и показать главный экран

## Шаг 7: Настройка URL сервера

1. Запустите приложение на симуляторе или устройстве
2. В поле **URL сервера** введите IP-адрес вашего ноутбука:
   - Найдите IP-адрес: в терминале выполните `ifconfig` или `ipconfig`
   - Формат: `http://192.168.1.XXX:5000`
3. Нажмите Enter или кнопку распознавания

## Возможные проблемы и решения

### Проблема: "No such module 'UIKit'" или похожие ошибки

**Решение:**
1. Очистите проект: `⌘⇧K`
2. Удалите папку `DerivedData`:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData
   ```
3. Пересоберите проект: `⌘B`

### Проблема: Файлы не добавляются в проект

**Решение:**
1. Убедитесь, что вы перетаскиваете файлы в правильную папку (синюю иконку)
2. Проверьте, что в диалоге добавления выбран правильный target

### Проблема: Приложение не запрашивает разрешения камеры

**Решение:**
1. Проверьте, что Info.plist содержит `NSCameraUsageDescription`
2. Убедитесь, что Info.plist подключён в настройках проекта
3. Удалите приложение с симулятора/устройства и переустановите

### Проблема: Ошибка "Cannot find type 'PredictionResponse' in scope"

**Решение:**
1. Убедитесь, что файл `PredictionResponse.swift` добавлен в проект
2. Проверьте, что он находится в папке `Models/`
3. Очистите и пересоберите проект

## Дополнительные настройки (опционально)

### Добавление Assets.xcassets

Если вы хотите добавить иконки и изображения:

1. В Xcode: File → New → File
2. Выберите **Asset Catalog**
3. Назовите `Assets`
4. Добавьте иконки и изображения по необходимости

### Настройка схемы для отладки

1. Product → Scheme → Edit Scheme...
2. Выберите **Run**
3. Настройте параметры отладки по необходимости

## Структура проекта после настройки

```
DigitRecognition/
├── DigitRecognition.xcodeproj/
├── DigitRecognition/
│   ├── Models/
│   │   └── PredictionResponse.swift
│   ├── Services/
│   │   ├── APIClient.swift
│   │   └── CameraService.swift
│   ├── Views/
│   │   ├── MainView.swift
│   │   └── ResultView.swift
│   ├── DigitRecognitionApp.swift
│   └── Info.plist
└── README.md (опционально)
```

## Следующие шаги

После настройки Xcode проекта:

1. Запустите Python сервер: `cd server && python app.py`
2. Запустите iOS приложение в Xcode
3. Введите URL сервера в приложении
4. Протестируйте распознавание цифр

---

**Примечание:** Если у вас возникнут проблемы, проверьте логи Xcode (View → Debug Area → Activate Console) для получения подробной информации об ошибках.
