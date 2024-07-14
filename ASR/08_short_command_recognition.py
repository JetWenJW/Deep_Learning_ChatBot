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

# 任選一檔案測試，發音為 happy
train_audio_path = './GoogleSpeechCommandsDataset/data/'
data, sr = librosa.load(train_audio_path+'happy/0ab3b47d_nohash_0.wav')

# 繪製波形
librosa.display.waveplot(data, sr)
print(data.shape)

# 任選一檔案測試，發音為 happy
train_audio_path = './GoogleSpeechCommandsDataset/data/'
data, sr = librosa.load(train_audio_path+'happy/0b09edd3_nohash_0.wav')

# 繪製波形
librosa.display.waveplot(data, sr)
print(data.shape)

# 播放
from IPython.display import Audio

Audio(data, rate=sr)

# 取得音檔的屬性
import wave

wav_file = train_audio_path+'happy/0ab3b47d_nohash_0.wav'
f=wave.open(wav_file)
print(f'取樣頻率={f.getframerate()}, 幀數={f.getnframes()}, ' +
      f'聲道={f.getnchannels()}, 精度={f.getsampwidth()}, ' +
      f'檔案秒數={f.getnframes() / f.getframerate():.2f}')
f.close()

nchannels2 = f.getnchannels()
sample_rate2 = f.getframerate()
sample_width2 = f.getsampwidth()

# 重抽樣，統一取 8000 個樣本
wav_file = train_audio_path+'happy/0ab3b47d_nohash_0.wav'
samples, sample_rate = librosa.load(wav_file, sr=None, res_type='kaiser_fast')
print(f'original sample rate={sample_rate}')
print(f'幀數={len(samples)}')
# 重抽樣，統一取 8000 個樣本
samples = librosa.resample(samples, sample_rate, 8000)
print(f'幀數={len(samples)}')

samples, sample_rate = librosa.load(wav_file, sr=None, res_type='kaiser_fast')
print(f'original sample rate={sample_rate}')
print(f'幀數={len(samples)}')
new_samples = np.pad(samples,(0, 16000-len(samples)),'constant')
print(f'幀數={len(new_samples)}')

# 重抽樣另一個檔，統一取 8000 個樣本
wav_file = train_audio_path+'bed/0d393936_nohash_0.wav'
samples, sample_rate = librosa.load(wav_file, sr=None, res_type='kaiser_fast')
print(f'original sample rate={sample_rate}')
print(f'幀數={len(samples)}')
# 重抽樣，統一取 8000 個樣本
samples = librosa.resample(samples, sample_rate, 8000)
print(f'幀數={len(samples)}')

samples[:100]

# 取得子目錄名稱
labels=os.listdir(train_audio_path)
labels

# 子目錄的檔案數
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
# 繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,6))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('指令', fontsize=12)
plt.ylabel('檔案數', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('子目錄的檔案數')
print(f'檔案數={no_of_recordings}')
plt.show()

# 載入音樂檔案
TOTAL_FRAME_COUNT = 16000 # 每個檔案統一的幀數
duration_of_recordings=[]
all_wave = []
y = []
for i, label in enumerate(labels):
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    class_wave=None
    for wav in waves:
        # 載入音樂檔案
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav
                                            ,sr=None , res_type='kaiser_fast')
        duration_of_recordings.append(float(len(samples)/sample_rate))
        # 長度不足，右邊補 0
        if len(samples) < TOTAL_FRAME_COUNT : 
            samples = np.pad(samples,(0, TOTAL_FRAME_COUNT-len(samples)),'constant')
        elif len(samples) > TOTAL_FRAME_COUNT : 
            samples = np.resize(samples, TOTAL_FRAME_COUNT)
            
        if class_wave is None:
            class_wave = samples.reshape(1, -1) 
        else:
            class_wave = np.concatenate([class_wave, samples.reshape(1, -1)], axis=0)
        y.append(i)
        
    all_wave.append(class_wave)    
    print(class_wave.shape)
    np.save('./GoogleSpeechCommandsDataset/' + label + '.npy', class_wave) # 存成 npy
fig = plt.hist(np.array(duration_of_recordings))

y[:10], y[-10:]

len(all_wave)

# 載入npy檔案
train_audio_path2 = './GoogleSpeechCommandsDataset/'
npy_files = [f for f in os.listdir(train_audio_path2) if f.endswith('.npy')]
print(npy_files)
all_wave = []
y = []
no=0
for i, label in enumerate(npy_files):
    class_wave = np.load(train_audio_path2+label)
    all_wave.append(class_wave)
    print(class_wave.shape)
    no+=class_wave.shape[0]
    y.extend(np.full(class_wave.shape[0], i)) 

# 計算 MFCC
MFCC_COUNT = 40
X = None
for class_wave in all_wave:
    for data in class_wave:
        mfcc = librosa.feature.mfcc(y=data, sr=len(data), n_mfcc=MFCC_COUNT)
        # print(data.shape, mfcc.shape)
        if X is None:
            X = mfcc.reshape(1, MFCC_COUNT, -1, 1)
        else:
            X = np.concatenate((X, mfcc.reshape(1, MFCC_COUNT, -1, 1)), axis=0)
    print(X.shape) 
print(X.shape, len(y))

np.save('./GoogleSpeechCommandsDataset/mfcc/mfcc.npy', X)
np.save('./GoogleSpeechCommandsDataset/mfcc/y.npy', y)
X = np.load('./GoogleSpeechCommandsDataset/mfcc/mfcc.npy')
y = np.load('./GoogleSpeechCommandsDataset/mfcc/y.npy')
X.shape, y.shape

# 資料切割
from sklearn.model_selection import train_test_split
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
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
        layers.Dense(len(labels), activation="softmax"),
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
fig = plt.legend(prop={"size":22})

# 預測函數
def predict(file_path):
    samples, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    # 繪製波形
    librosa.display.waveplot(samples, sr)
    plt.show()
    
    # 右邊補 0
    if len(samples) < TOTAL_FRAME_COUNT : 
        samples = np.pad(samples,(0, TOTAL_FRAME_COUNT-len(samples)),'constant')
    elif len(samples) > TOTAL_FRAME_COUNT : 
        # 取中間一段
        oversize = len(samples) - TOTAL_FRAME_COUNT
        samples = samples[int(oversize/2):int(oversize/2)+TOTAL_FRAME_COUNT]

    # 繪製波形
    librosa.display.waveplot(samples, sr)
    plt.show()

    # 驗證 mfcc 是否需要標準化
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=MFCC_COUNT)
    for i in range(mfcc.shape[1]):
        plt.scatter(x=range(mfcc.shape[0]), y=mfcc[:, i].reshape(-1))
    X_pred = mfcc.reshape(1, *mfcc.shape, 1)
    
    print(X_pred.shape, samples.shape)
    # 預測
    prob = model.predict(X_pred)
    return np.around(prob, 2), labels[np.argmax(prob)]
# 任選一檔案測試，該檔案發音為 bed
train_audio_path = './GoogleSpeechCommandsDataset/data/'
predict(train_audio_path+'bed/0d2bcf9d_nohash_0.wav')

# 任選一檔案測試，該檔案發音為 cat
predict(train_audio_path+'cat/0ac15fe9_nohash_0.wav')

# 任選一檔案測試，該檔案發音為 happy
predict(train_audio_path+'happy/0ab3b47d_nohash_0.wav')


# 自行使用 14_10_record.py 錄音，指令：
# python 14_10_record.py GoogleSpeechCommandsDataset/happy.wav
# 測試，該檔案發音為 happy
predict('./GoogleSpeechCommandsDataset/happy.wav')

# 測試，該檔案發音為 cat
predict('./GoogleSpeechCommandsDataset/cat.wav')

# 測試，該檔案發音為 bed
predict('./GoogleSpeechCommandsDataset/bed.wav')