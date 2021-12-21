# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers.models.speech_to_text import Speech2TextTokenizer
import pandas as pd
import numpy as np
import re, nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
nltk.download('stopwords')
from bert_dataset import CustomDataset
from bert_classifier import BertClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


files.upload()

data = pd.read_excel('data_ver00.xlsx')
data

data.loc[data.Information > 0.0, 'Information'] = 1.0
data.loc[data.Emotion > 0.0, 'Emotion'] = 1.0
data.loc[data.Action > 0.0, 'Action'] = 1.0
data.loc[(np.isnan(data.Information)) & (~(np.isnan(data.Emotion) & np.isnan(data.Action))), 'Information'] = 0.0
data.loc[(np.isnan(data.Emotion)) & (~(np.isnan(data.Information) & np.isnan(data.Action))), 'Emotion'] = 0.0
data.loc[(np.isnan(data.Action)) & (~(np.isnan(data.Emotion) & np.isnan(data.Information))), 'Action'] = 0.0

data = data.dropna()
data = data.reset_index(drop=True)
data

data_train, data_test = \
              np.split(data.sample(frac=1, random_state=42), 
                       [int(0.6*len(data))])

pattern = "[^А-Яа-я-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def preprocessing(text):
    text = re.sub(pattern, ' ', text)
    res = ''
    k=0
    for token in text.split():
          k+=1
          token = token.strip('-') # убирает тире в начале слова
          if (k<200):
            if (token != ''):                                  
              res = (' '.join([res, token])).lower().strip()
          else:
            break     
    if (res != ""):
      return res

text_train = data_train['text'].apply(preprocessing)
class_train = data_train[['Information','Emotion', 'Action']].astype(int)
data_train = (pd.concat([text_train, class_train], axis = 1)).dropna()

text_test = data_test['text'].apply(preprocessing)
class_test = data_test[['Information', 'Emotion', 'Action']].astype(int)
data_test = (pd.concat([text_test, class_test], axis = 1)).dropna()

#Information

classifier_inf = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=2,
        epochs=5,
        model_save_path='/content/bert.pt'
)

classifier_inf.preparation(
        X_train=list(data_train['text']),
        y_train=list(data_train['Information']),
        X_valid=list(data_test['text']),
        y_valid=list(data_test['Information'])
    )

classifier_inf.train()

texts = list(data_test['text'])
labels = list(data_test['Information'])

predictions = [classifier_inf.predict(t) for t in texts]


precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
accuracy = accuracy_score(labels, predictions)

print(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}')

#Emotion

classifier_emo = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=2,
        epochs=5,
        model_save_path='/content/bert.pt'
)

classifier_emo.preparation(
        X_train=list(data_train['text']),
        y_train=list(data_train['Emotion']),
        X_valid=list(data_test['text']),
        y_valid=list(data_test['Emotion'])
    )

classifier_emo.train()

texts = list(data_test['text'])
labels = list(data_test['Emotion'])

predictions = [classifier_emo.predict(t) for t in texts]

precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
accuracy = accuracy_score(labels, predictions)

print(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}')

#Action

classifier_act = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=2,
        epochs=5,
        model_save_path='/content/bert.pt'
)

classifier_act.preparation(
        X_train=list(data_train['text']),
        y_train=list(data_train['Action']),
        X_valid=list(data_test['text']),
        y_valid=list(data_test['Action'])
    )

classifier_act.train()

texts = list(data_test['text'])
labels = list(data_test['Action'])

predictions = [classifier_act.predict(t) for t in texts]

precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
accuracy = accuracy_score(labels, predictions)

print(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}')