# 載入相關套件
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from python_speech_features import mfcc, logfbank

# 載入音樂檔案
sr, data = wavfile.read("./audio/WAV_1MG.wav")

# 讀取 MFCC、Filter bank 特徵
mfcc_features = mfcc(data, sr)
filterbank_features = logfbank(data, sr)

# Print parameters
print('MFCC 維度:', mfcc_features.shape)
print('Filter bank 維度:', filterbank_features.shape)

mfcc_features[0]

# 繪圖
plt.subplot(2,1,1)
mfcc_features = mfcc_features.T
plt.imshow(mfcc_features, cmap=plt.cm.jet, 
    extent=[0, mfcc_features.shape[1], 0, mfcc_features.shape[0]], aspect='auto')
plt.title('MFCC')

plt.subplot(2,1,2)
filterbank_features = filterbank_features.T
plt.imshow(filterbank_features, cmap=plt.cm.jet, 
   extent=[0, filterbank_features.shape[1], 0, filterbank_features.shape[0]], aspect='auto')
plt.title('Filter bank')
plt.tight_layout()
plt.show()


