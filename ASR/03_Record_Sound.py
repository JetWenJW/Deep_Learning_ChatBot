# 載入相關套件
import speech_recognition as sr
import pyttsx3

# 列出電腦中的說話者(Speaker)
speak = pyttsx3.init()
voices = speak.getProperty('voices')
for voice in voices: 
    print("Voice:") 
    print(" - ID: %s" % voice.id) 
    print(" - Name: %s" % voice.name) 
    print(" - Languages: %s" % voice.languages) 
    print(" - Gender: %s" % voice.gender) 
    print(" - Age: %s" % voice.age)

# 指定說話者
speak.setProperty('voice', voices[1].id)

# https://pyttsx3.readthedocs.io/en/latest/engine.html
speak.setProperty('rate', 150)

speak.say('No One Wants 40 Hour Work Weeks Anymore. Everyone Wants to Work 4 Hours Per Week on a Laptop In Bali')
# 等待說完
speak.runAndWait()

# 指定說話者
speak.setProperty('voice', voices[0].id)

speak.say('受峰面影響北台灣今天下午大雨特報，有些道路甚至發生積淹，曾文水庫上游也傳來好消息')
# 等待說完
speak.runAndWait()

# 麥克風收音
# 受峰面影響北台灣今天下午大雨特報，有些道路甚至發生淹水
r = sr.Recognizer()
with sr.Microphone() as source:
    # 文字轉語音
    speak.say('請說話...')
    # 等待說完
    speak.runAndWait()
    
    #降噪
    r.adjust_for_ambient_noise(source)
    # 麥克風收音
    audio = r.listen(source)


# 語音辨識
# 受峰面影響北台灣今天下午大雨特報有，些道路甚至發生積淹，曾文水庫上游也傳來好消息
try:
    text=r.recognize_google(audio, language='zh-tw')
    print(text)
except e:
    pass

# 錄音存檔    
wav_file = "./audio/woman.wav"
with open(wav_file, "wb") as f:
    f.write(audio.get_wav_data(convert_rate=16000))

import IPython

# autoplay=True：自動播放，不須按 PLAY 鍵
IPython.display.Audio(wav_file, autoplay=True) 

# 取得音檔的屬性
import wave

f=wave.open(wav_file)
print(f'取樣頻率={f.getframerate()}, 幀數={f.getnframes()}, ' +
      f'聲道={f.getnchannels()}, 精度={f.getsampwidth()}, ' +
      f'檔案秒數={f.getnframes() / (f.getframerate() * f.getnchannels()):.2f}')
f.close()

import speech_recognition as sr

# 讀取音檔，轉為音訊
r = sr.Recognizer()
with sr.WavFile(wav_file) as source:
    audio = r.record(source)
    
# 語音辨識
try:
    text=r.recognize_google(audio, language='zh-tw')
    print(text)
except e:
    pass

# 顯示所有可能的辨識結果及信賴度
dict1=r.recognize_google(audio, show_all=True, language='zh-tw')
for i, item in enumerate(dict1['alternative']):
    if i == 0:
        print(f"信賴度={item['confidence']}\n{item['transcript']}")
    else:
        print(f"{item['transcript']}")