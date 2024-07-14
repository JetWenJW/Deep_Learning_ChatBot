# 載入相關套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import pathlib
import csv
import tensorflow as tf
from tensorflow.keras import layers

# 不顯示警告訊息
import warnings
warnings.filterwarnings('ignore')

# 載入音樂檔案
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

X = None
y = []
for i, g in enumerate(genres):
    pathlib.Path(f'./GTZAN/genres//{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./GTZAN/genres/{g}'):
        songname = f'./GTZAN/genres/{g}/{filename}'
        data, sr = librosa.load(songname, mono=True, duration=25)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        # print(data.shape, mfcc.shape)
        if X is None:
            X = mfcc.reshape(1, 40, -1, 1)
        else:
            X = np.concatenate((X, mfcc.reshape(1, 40, -1, 1)), axis=0)
        y.append(i)
        
print(X.shape, len(y))
X[0]
X.min(axis=0).shape

# 常態化
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 資料切割
from sklearn.model_selection import train_test_split
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.2)
X_train.shape, X_test.shape

# CNN 模型
input_shape = X_train.shape[1:]
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# CNN 模型
input_shape = X_train.shape[1:]
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型訓練
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# 評分(Score Model)
score=model.evaluate(X_test, y_test, verbose=0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')


# 對訓練過程的準確率繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], 'r', label='訓練準確率')
plt.plot(history.history['val_accuracy'], 'g', label='驗證準確率')
plt.legend()

# 載入音樂檔案
X = None
y = []
for i, g in enumerate(genres):
    pathlib.Path(f'./GTZAN/genres//{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./GTZAN/genres/{g}'):
        songname = f'./GTZAN/genres/{g}/{filename}'
        data, sr = librosa.load(songname, mono=True, duration=25)
        try:
            if i == 0:
                segment_length = int(data.shape[0] / 10)
            for j in range(10):
                segment = data[j * segment_length: (j+1) * segment_length]
                # print(segment.shape)
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
                # print(data.shape, mfcc.shape)
                if X is None:
                    X = mfcc.reshape(1, 40, -1, 1)
                else:
                    X = np.concatenate((X, mfcc.reshape(1, 40, -1, 1)), axis=0)
                y.append(i)
        except:
            print(i)
            raise Exception('')
print(X.shape, len(y))

# 常態化
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 資料切割
from sklearn.model_selection import train_test_split
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.2)
X_train.shape, X_test.shape

# CNN 模型
input_shape = X_train.shape[1:]
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型訓練
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# 評分(Score Model)
score=model.evaluate(X_test, y_test, verbose=0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')

# 對訓練過程的準確率繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], 'r', label='訓練準確率')
plt.plot(history.history['val_accuracy'], 'g', label='驗證準確率')
plt.legend()