import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image
from sklearn.utils import shuffle
import json
import h5py
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    accuracy_score,
)
import tensorflow as tf


# class DigitStructWrapper:
#     def __init__(self, inf):
#         self.inf = h5py.File(inf, 'r')
#         self.digitStructName = self.inf['digitStruct']['name']
#         self.digitStructBbox = self.inf['digitStruct']['bbox']

#     def get_name(self, n):
#         """Return the name of the n(th) digit struct"""
#         return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]][()]])

#     def get_attribute(self, attr):
#         """Helper function for dealing with one vs. multiple bounding boxes"""
#         if (len(attr) > 1):
#             attr = [self.inf[attr[j].item()][0][0] for j in range(len(attr))]
#         else:
#             attr = [attr[0][0]]
#         return attr

#     def get_bbox(self, n):
#         """Return a dict containing the data from the n(th) bbox"""
#         bbox = {}
#         bb = self.digitStructBbox[n].item()
#         bbox['height'] = self.get_attribute(self.inf[bb]["height"])
#         bbox['label'] = self.get_attribute(self.inf[bb]["label"])
#         bbox['left'] = self.get_attribute(self.inf[bb]["left"])
#         bbox['top'] = self.get_attribute(self.inf[bb]["top"])
#         bbox['width'] = self.get_attribute(self.inf[bb]["width"])
#         return bbox

#     def get_item(self, n):
#         """Return the name and bounding boxes of a single image"""
#         s = self.get_bbox(n)
#         s['name'] = self.get_name(n)
#         return s

#     def unpack(self):
#         """Returns a list of dicts containing all the bounding boxes"""
#         return [self.get_item(i) for i in range(len(self.digitStructName))]

#     def unpack_all(self):
#         pictDat = self.unpack()
#         result = []
#         structCnt = 1
#         for i in range(len(pictDat)):
#             item = {'filename': pictDat[i]["name"]}
#             figures = []
#             for j in range(len(pictDat[i]['height'])):
#                 figure = {}
#                 figure['height'] = pictDat[i]['height'][j]
#                 figure['label'] = pictDat[i]['label'][j]
#                 figure['left'] = pictDat[i]['left'][j]
#                 figure['top'] = pictDat[i]['top'][j]
#                 figure['width'] = pictDat[i]['width'][j]
#                 figures.append(figure)
#             structCnt = structCnt + 1
#             item['boxes'] = figures
#             result.append(item)
#         return result


# def parse_info(start_path):
#     """ Extracts a bounding box file and returns a dictionary
#     """
#     return DigitStructWrapper(start_path).unpack_all()

# # Extract info from mat files such as the bounding boxes
# train_bbox = parse_info('/home/vadzim/train/digitStruct.mat')
# test_bbox = parse_info('/home/vadzim/test/digitStruct.mat')
# extra_bbox = parse_info('/home/vadzim/extra/digitStruct.mat')

# # Display the information stored about an individual image
# print(json.dumps(train_bbox[0], indent=2))


# # Select an image and the corresponding boxes
# image = '/home/vadzim/train/30485.png'
# image_bounding_boxes = train_bbox[30484]['boxes']


# def dict_to_df(image_bounding_boxes, path):
#     """ Helper function for flattening the bounding box dictionary
#     """
#     # Store each bounding box
#     boxes = []

#     # For each set of bounding boxes
#     for image in image_bounding_boxes:
#         # For every bounding box
#         for bbox in image['boxes']:
            
#             # Store a dict with the file and bounding box info
#             boxes.append({
#                     'filename': path + image['filename'],
#                     'label': bbox['label'],
#                     'width': bbox['width'],
#                     'height': bbox['height'],
#                     'top': bbox['top'],
#                     'left': bbox['left']})

#     # return the data as a DataFrame
#     return pd.DataFrame(boxes)


# # Save bounding box data to csv
# bbox_file = '/home/vadzim/bounding_boxes.csv'

# if not os.path.isfile(bbox_file):
#     # Extract every individual bounding box as DataFrame  
#     train_df = dict_to_df(train_bbox, '/home/vadzim/train/')
#     test_df = dict_to_df(test_bbox, '/home/vadzim/test/')
#     extra_df = dict_to_df(extra_bbox, '/home/vadzim/extra/')

#     train_df['dataset'] = 'train'
#     test_df['dataset'] = 'test'
#     extra_df['dataset'] = 'extra'

#     print("Training", train_df.shape)
#     print("Test", test_df.shape)
#     print("Extra", extra_df.shape)
#     print('')

#     # Concatenate all the information in a single file
#     df = pd.concat([train_df, test_df, extra_df])
    
#     print("Combined", df.shape)

#     # Write dataframe to csv
#     df.to_csv(bbox_file, index=False)

#     # Delete the old dataframes to save memory
#     del train_df, test_df, extra_df, train_bbox, test_bbox, extra_bbox
# else:
#     # Load preprocessed bounding boxes
#     df = pd.read_csv(bbox_file)

# # Rename the columns to more suitable names
# df.rename(columns={'left': 'x0', 'top': 'y0'}, inplace=True)

# # Calculate x1 and y1
# df['x1'] = df['x0'] + df['width']
# df['y1'] = df['y0'] + df['height']

# # Perform the following aggregations on the columns
# aggregate = {
#     'x0': ('x0', 'min'),
#     'y0': ('y0', 'min'),
#     'x1': ('x1', 'max'),
#     'y1': ('y1', 'max'),
#     'labels': ('label', list),
#     'num_digits': ('label', 'count'),
# }

# # Apply the aggegration
# df = df.groupby('filename').agg(**aggregate).reset_index()

# # Fix the column names after aggregation
# # df.columns = [x[0] if i < 5 else x[1] for i, x in enumerate(df.columns.values)]

# # Calculate the increase in both directions
# df['x_inc'] = ((df['x1'] - df['x0']) * 0.30) / 2.
# df['y_inc'] = ((df['y1'] - df['y0']) * 0.30) / 2.

# # Apply the increase in all four directions
# df['x0'] = (df['x0'] - df['x_inc']).astype('int')
# df['y0'] = (df['y0'] - df['y_inc']).astype('int')
# df['x1'] = (df['x1'] + df['x_inc']).astype('int')
# df['y1'] = (df['y1'] + df['y_inc']).astype('int')


# def get_img_size(filepath):
#     """Returns the image size in pixels given as a 2-tuple (width, height)
#     """
#     image = Image.open(filepath)
#     return image.size 

# def get_img_sizes(folder):
#     """Returns a DataFrame with the file name and size of all images contained in a folder
#     """
#     image_sizes = []

#     # Get all .png images contained in the folder
#     images = [img for img in os.listdir(folder) if img.endswith('.png')]

#     # Get image size of every individual image
#     for image in images:
#         w, h = get_img_size(folder + image)
#         image_size = {'filename': folder + image, 'image_width': w, 'image_height': h}
#         image_sizes.append(image_size)
#     # Return results as a pandas DataFrame
#     return pd.DataFrame(image_sizes)


# # Extract the image sizes
# train_sizes = get_img_sizes('/home/vadzim/train/')
# test_sizes = get_img_sizes('/home/vadzim/test/')
# extra_sizes = get_img_sizes('/home/vadzim/extra/')

# # Concatenate all the information in a single file
# image_sizes = pd.concat([train_sizes, test_sizes, extra_sizes])

# # Delete old dataframes
# del train_sizes, test_sizes, extra_sizes

# print("Bounding boxes", df.shape)
# print("Image sizes", image_sizes.shape)
# print('')

# print(df['filename'], image_sizes['filename'])

# # Inner join the datasets on filename
# df = pd.merge(df, image_sizes, on='filename', how='inner')

# print("Combined", df.shape)

# # Delete the image size df
# del image_sizes

# # Store checkpoint
# df.to_csv("/home/vadzim/image_data.csv", index=False)
# #df = pd.read_csv('data/image_data.csv')

# df.loc[df['x0'] < 0, 'x0'] = 0
# df.loc[df['y0'] < 0, 'y0'] = 0
# df.loc[df['x1'] > df['image_width'], 'x1'] = df['image_width']
# df.loc[df['y1'] > df['image_height'], 'y1'] = df['image_height']

# # Count the number of images by number of digits
# df.num_digits.value_counts(sort=False)

# # Keep only images with less than 6 digits
# df = df[df.num_digits < 6]

# df[['image_width', 'image_height']].describe()

# print("Bounding boxes", df.shape)
# print('')

# def crop_and_resize(image, img_size):
#     """ Crop and resize an image """
#     # Открываем картинку
#     image_data = Image.open(image['filename'])
    
#     # Обрезаем. PIL ожидает кортеж (left, top, right, bottom)
#     crop = image_data.crop((image['x0'], image['y0'], image['x1'], image['y1']))
    
#     # Изменяем размер
#     resized = crop.resize(img_size)
    
#     # Возвращаем как numpy массив, так как мы будем класть это в массив X
#     return np.array(resized)


# def create_dataset(df, img_size):
#     """ Helper function for converting images into a numpy array
#     """
#     # Initialize the numpy arrays (0's are stored as 10's)
#     X = np.zeros(shape=(df.shape[0], img_size[0], img_size[0], 3), dtype='uint8')
#     y = np.full((df.shape[0], 5), 10, dtype=int)

#     # Iterate over all images in the pandas dataframe (slow!)
#     for i, (index, image) in enumerate(df.iterrows()):
#         # Get the image data
#         X[i] = crop_and_resize(image, img_size)

#         # Get the label list as an array
#         labels = np.array((image['labels']))

#         # Store 0's as 0 (not 10)
#         labels[labels==10] = 0

#         # Embed labels into label array
#         y[i,0:labels.shape[0]] = labels

#     # Return data and labels   
#     return X, y


# # Change this to select a different image size
# image_size = (32, 32)

# print("Shape of df train:", df[df.filename.str.contains('train')].shape)

# # Get cropped images and labels (this might take a while...)
# X_train, y_train = create_dataset(df[df.filename.str.contains('train')], image_size)
# X_test, y_test = create_dataset(df[df.filename.str.contains('test')], image_size)
# X_extra, y_extra = create_dataset(df[df.filename.str.contains('extra')], image_size)

# # We no longer need the dataframe
# del df

# print("Training", X_train.shape, y_train.shape)
# print("Test", X_test.shape, y_test.shape)
# print('Extra', X_extra.shape, y_extra.shape)

# # Train set
# y_train_counts = np.unique((y_train != 10).sum(1), return_counts=True)

# y_train_counts = list(zip(y_train_counts[0], y_train_counts[1]))
# y_train_df_counts = pd.DataFrame(y_train_counts,  columns= ['Number of Digits', 'Count'])
# y_train_df_counts.set_index('Number of Digits', inplace=True)

# # Test set
# y_test_counts = np.unique((y_test != 10).sum(1), return_counts=True)

# y_test_counts = list(zip(y_test_counts[0], y_test_counts[1]))
# y_test_df_counts = pd.DataFrame(y_test_counts, columns= ['Number of Digits', 'Count'])
# y_test_df_counts.set_index('Number of Digits', inplace=True)

# # Extra set
# y_extra_counts = np.unique((y_extra != 10).sum(1), return_counts=True)

# y_extra_counts = list(zip(y_extra_counts[0], y_extra_counts[1]))
# y_extra_df_counts = pd.DataFrame(y_extra_counts, columns= ['Number of Digits', 'Count'])
# y_extra_df_counts.set_index('Number of Digits', inplace=True)

# combined_counts_df = pd.concat([y_train_df_counts, y_test_df_counts, y_extra_df_counts], 
#                                keys=['Train', 'Test', 'Extra'],
#                               names=['Dataset'])
# print(combined_counts_df)

# # Initialize the subplots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(16, 4))

# # Set the main figure title
# fig.suptitle('Number of digits per image distribution', fontsize=14, fontweight='bold', y=1.05)

# def random_sample(N, K):
#     """Return a boolean mask of size N with K selections
#     """
#     mask = np.array([True]*K + [False]*(N-K))
#     np.random.shuffle(mask)
#     return mask

# # Pick 8000 training and 2000 extra samples
# sample1 = random_sample(X_train.shape[0], 8000)
# sample2 = random_sample(X_extra.shape[0], 2000)

# # Create valdidation from the sampled data
# X_val = np.concatenate([X_train[sample1], X_extra[sample2]])
# y_val = np.concatenate([y_train[sample1], y_extra[sample2]])

# # Keep the data not contained by sample
# X_train = np.concatenate([X_train[~sample1], X_extra[~sample2]])
# y_train = np.concatenate([y_train[~sample1], y_extra[~sample2]])

# # Moved to validation and training set
# # del X_extra, y_extra 

# print("Training", X_train.shape, y_train.shape)
# print('Validation', X_val.shape, y_val.shape)

# # Create file
# h5f = h5py.File('/home/vadzim/multi_digit_rgb.h5', 'w')

# # Store the datasets
# h5f.create_dataset('X_train', data=X_train)
# h5f.create_dataset('y_train', data=y_train)
# h5f.create_dataset('X_test', data=X_test)
# h5f.create_dataset('y_test', data=y_test)
# h5f.create_dataset('X_val', data=X_val)
# h5f.create_dataset('y_val', data=y_val)

# # Close the file
# h5f.close()

# def rgb2gray(images):
#     return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

# # Transform the images to greyscale
# X_train = rgb2gray(X_train).astype(np.float32)
# X_test = rgb2gray(X_test).astype(np.float32)
# X_val = rgb2gray(X_val).astype(np.float32)

# # Calculate the mean on the training data
# train_mean = np.mean(X_train, axis=0)

# # Calculate the std on the training data
# train_std = np.std(X_train, axis=0)

# # Subtract it equally from all splits
# train_norm = (X_train - train_mean) / train_std
# test_norm = (X_test - train_mean)  / train_std
# val_norm = (X_val - train_mean) / train_std

# # Create file
# h5f = h5py.File('/home/vadzim/multi_digit_norm_grayscale.h5', 'w')

# # Store the datasets
# h5f.create_dataset('X_train', data=train_norm)
# h5f.create_dataset('y_train', data=y_train)
# h5f.create_dataset('X_test', data=test_norm)
# h5f.create_dataset('y_test', data=y_test)
# h5f.create_dataset('X_val', data=val_norm)
# h5f.create_dataset('y_val', data=y_val)

# # Close the file
# h5f.close()

# Model learning

# Open the HDF5 file containing the datasets
with h5py.File('/home/vadzim/multi_digit_norm_grayscale.h5','r') as h5f:
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]


print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

# Get the image data information & dimensions
train_count, img_height, img_width, num_channels = X_train.shape

# Get label information
num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))


# --- 1. ПАРАМЕТРЫ И КОНСТАНТЫ ---
img_height, img_width, num_channels = 32, 32, 1
num_labels = 11  # 10 цифр + 1 пустая заглушка

epochs = 100
batch_size = 512
display_step = 200

# Коэффициенты Dropout (в Keras передается rate = 1 - keep_prob)
dropout_rate_conv = 0.50  # 1 - 0.50
dropout_rate_fc = 0.50    # 1 - 0.50

CHECKPOINT_PATH = 'checkpoints'
LOG_DIR = 'logs'

# Путь к файлу весов
checkpoint_file = os.path.join(CHECKPOINT_PATH, 'svhn_model.weights.h5')

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


# --- 2. КАСТОМНАЯ МЕТРИКА ТОЧНОСТИ ---
@tf.keras.utils.register_keras_serializable()
def sequence_accuracy(y_true, y_pred):
    """
    Вычисляет точность всей последовательности. 
    Строка считается верной, только если ВСЕ 5 цифр угаданы правильно.
    """
    pred_cls = tf.argmax(y_pred, axis=-1)  # Форма: (batch_size, 5)
    y_true = tf.cast(y_true, tf.int64)
    
    # Сравниваем поэлементно
    correct_per_digit = tf.cast(tf.equal(y_true, pred_cls), tf.float32)
    # Ищем минимум по строке: если есть хоть один 0, вся строка будет 0
    correct_sequence = tf.reduce_min(correct_per_digit, axis=-1)
    
    return tf.reduce_mean(correct_sequence) * 100.0


# --- 3. СБОРКА МОДЕЛИ (Functional API) ---
def build_svhn_model(initializer='xavier'):
    # Задаем базовый инициализатор Keras
    keras_init = 'glorot_uniform' if initializer == 'xavier' else 'he_normal'
    
    # Входной тензор
    inputs = tf.keras.Input(shape=(img_height, img_width, num_channels), name='x')
    
    # Conv Block 1
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate_conv)(x)
    
    # Conv Block 2
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_4')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate_conv)(x)
    
    # Conv Block 3
    x = tf.keras.layers.Conv2D(128, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_5')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    
    x = tf.keras.layers.Conv2D(128, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_6')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    
    x = tf.keras.layers.Conv2D(128, (5, 5), padding='same', kernel_initializer=keras_init, name='conv_7')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate_fc)(x)
    
    # Сглаживание (вместо кастомного flatten_layer)
    x = tf.keras.layers.Flatten()(x)
    
    # Fully-connected 1
    x = tf.keras.layers.Dense(256, kernel_initializer=keras_init, name='fc_1')(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    x = tf.keras.layers.Dropout(dropout_rate_fc)(x)
    
    # Fully-connected 2
    x = tf.keras.layers.Dense(256, kernel_initializer=keras_init, name='fc_2')(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.10)(x)
    
    # Элегантный аналог 5 параллельных слоев:
    # Проецируем выходы сразу в размерность (5 * 11) и меняем форму тензора
    outputs = tf.keras.layers.Dense(num_digits * num_labels, kernel_initializer=keras_init)(x)
    outputs = tf.keras.layers.Reshape((num_digits, num_labels), name='y_pred')(outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='SVHN_Model')

model = build_svhn_model(initializer='xavier')


# --- 4. ОПТИМИЗАТОР И ОБУЧЕНИЕ ---

# График падения скорости обучения (Learning Rate Decay)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=8800,
    decay_rate=0.5,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Компиляция (связываем лосс, метрику и оптимизатор вместе)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[sequence_accuracy]
)


# --- 5. ВОССТАНОВЛЕНИЕ ЧЕКПОИНТА ---
if os.path.exists(checkpoint_file):
    print("Attempting to restore from last checkpoint ...")
    try:
        model.load_weights(checkpoint_file)
        print("Restored checkpoint successfully from:", checkpoint_file)
    except Exception as e:
        print("Failed to restore checkpoint, training from scratch. Error:", e)
else:
    print("No checkpoint found - initializing variables and training from scratch")


# --- 6. ЗАПУСК ОБУЧЕНИЯ И ВАЛИДАЦИИ ---
# Колбэки заменяют кастомное логирование, tf.summary и saver.save()
callbacks = [
    # Ведение логов для TensorBoard (включая автоматические гистограммы весов)
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    # Автоматическое сохранение весов модели в конце каждой эпохи
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        save_best_only=False
    )
]

print('\n=====================================================')
print('Starting Training Pipeline...')
print('=====================================================')

start_time = time.time()

# Метод fit() сам берет на себя циклы по эпохам, батчам, шаффлинг и валидацию
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    shuffle=True,  # Перемешивание данных перед каждой эпохой
    callbacks=callbacks
)

time_diff = time.time() - start_time
print('\n=====================================================')
print("Total training time usage: " + str(timedelta(seconds=int(round(time_diff)))))

# Финальная оценка на тестовом датасете
print("Evaluating on test set...")
test_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
# test_results[0] - это loss, test_results[1] - это наша метрика sequence_accuracy
print("Final test accuracy: %.4f" % test_results[1])
print('=====================================================')

# Генерируем сырые предсказания (логиты) для всего тестового набора
# Dropout отключается автоматически!
raw_predictions = model.predict(X_test, batch_size=512)

# raw_predictions имеет форму (batch_size, 5, 11).
# Нам нужно получить индексы классов (от 0 до 10) для каждой из 5 цифр.
# Берем argmax по последней оси и конвертируем тензор обратно в numpy-массив.
test_pred = tf.argmax(raw_predictions, axis=-1).numpy()

# Display the predictions
print(test_pred)

print(test_pred.shape)

def calculate_accuracy(a, b):
    """ Calculating the % of similar rows in two numpy arrays 
    """
    # Compare two numpy arrays row-wise
    correct = np.sum(np.all(a == b, axis=1))
    return 100.0 * (correct / float(a.shape[0]))


total_acc = calculate_accuracy(test_pred, y_test)

print('Multiple Digit Test Accuracy: %.3f %%' % total_acc)

# Find the position of the non missing labels
non_zero = np.where(y_test.flatten() != 10)

# Calculate the accuracy on the individual digit level
ind_acc = accuracy_score(test_pred.flatten()[non_zero], y_test.flatten()[non_zero]) * 100.0

print('Individual Digit Test Accuracy: %.3f %%' % ind_acc)

# Calculate the confusion matrix
cm = confusion_matrix(y_test.flatten()[non_zero], test_pred.flatten()[non_zero])

# Normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

f1 = f1_score(test_pred.flatten()[non_zero], y_test.flatten()[non_zero], average='weighted')

print('Individual Digit F1 Score: %.4f' % f1)

cls_report = classification_report(test_pred.flatten()[non_zero], y_test.flatten()[non_zero], digits=4)
print(cls_report)

# For every possible sequence length
for num_digits in range(1, 6):
    # Find all images with that given sequence length (returns an boolean array of True & Falses)
    images = np.where((y_test != 10).sum(1) == num_digits)

    # Calculate the accuracy on those images
    acc = calculate_accuracy(test_pred[images], y_test[images])

    print("%d digit accuracy %.3f %%" % (num_digits, acc))

# Find the correctly classified examples
correct = np.array([(a==b).all() for a, b in zip(test_pred, y_test)])

# Select the incorrectly classified examples
images = X_test[correct]
cls_true = y_test[correct]
cls_pred = test_pred[correct]

# Find the incorrectly classified examples
incorrect = np.invert(correct)

# Select the incorrectly classified examples
images = X_test[incorrect]
cls_true = y_test[incorrect]
cls_pred = test_pred[incorrect]
