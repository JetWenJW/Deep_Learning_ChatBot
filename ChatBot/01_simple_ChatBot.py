# 載入相關套件
import spacy
import json
import random
import pandas as pd

# 訓練資料
data_file = open('./chatbot_data/intents.json').read()
intents = json.loads(data_file)

intent_list = []
documents = []
responses = []

# 讀取所有意圖、例句、回應
for i, intent in enumerate(intents['intents']):
    # 例句
    for pattern in intent['patterns']:
        # adding documents
        documents.append((pattern, intent['tag'], i))

        # adding classes to our class list
        if intent['tag'] not in intent_list:
            intent_list.append(intent['tag'])
            
    # 回應(responses)
    for response in intent['responses']:
        responses.append((i, response))

responses_df = pd.DataFrame(responses, columns=['no', 'response'])
    
print(f'例句個數:{len(documents)}, intent個數:{len(intent_list)}')
responses_df

# 載入詞向量
nlp = spacy.load("en_core_web_md")

from spacy.lang.en.stop_words import STOP_WORDS

# 去除停用詞函數
def remove_stopwords(text1):
    filtered_sentence =[] 
    doc = nlp(text1)
    for word in doc:
        if word.is_stop == False: # 停用詞檢查
            filtered_sentence.append(word.lemma_) # lemma_：詞形還原
    return nlp(' '.join(filtered_sentence))

# 結束用語
def say_goodbye():
    tag = 1 # goodbye 項次
    response_filter = responses_df[responses_df['no'] == tag][['response']]
    selected_response = response_filter.sample().iloc[0, 0]
    return selected_response

# 結束用語
def say_not_understand():
    tag = 3 # 不理解的項次
    response_filter = responses_df[responses_df['no'] == tag][['response']]
    selected_response = response_filter.sample().iloc[0, 0]
    return selected_response


# 測試
prob_thread =0.6 # 相似度下限
while True:
    max_score = 0
    intent_no = -1
    similar_question = ''

    question = input('請輸入:\n')
    if question == '':
        break
        
    doc1 = remove_stopwords(question)
    
    # 比對：相似度比較
    for utterance in documents:
        # 兩語句的相似度比較
        doc2 = remove_stopwords(utterance[0])
        if len(doc1) > 0 and len(doc2) > 0:
            score = doc1.similarity(doc2)
            # print(utterance[0], score)
        # else:
            # print('\n', utterance[0],'\n')
            
        if score > max_score:
            max_score = score
            intent_no = utterance[2]
            similar_question = utterance[1] +', '+utterance[0]
    
    # 若找到相似問題，且高於相似度下限，才回答問題
    if intent_no == -1 or max_score < prob_thread:
        print(say_not_understand())
    else:
        print(f'你問的是：{similar_question}')
        response_filter = responses_df[responses_df['no'] == intent_no][['response']]
        # print(response_filter)
        selected_response = response_filter.sample().iloc[0, 0]
        # print(type(selected_response))
        print(f'回答：{selected_response}')

# say goodbye!        
print(f'回答：{say_goodbye()}')













