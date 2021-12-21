# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# %tensorflow_version 2.x

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, MaxPooling1D, Dropout, LSTM, Bidirectional, SpatialDropout1D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
nltk.download('stopwords')
import sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import keras_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam

# %matplotlib inline


data = pd.read_excel('data_ver00.xlsx')
data

# Делаю классификацию более общей
data.loc[data.Information > 0.0, 'Information'] = 1.0
data.loc[data.Emotion > 0.0, 'Emotion'] = 1.0
data.loc[data.Action > 0.0, 'Action'] = 1.0
data.loc[(np.isnan(data.Information)) & (~(np.isnan(data.Emotion) & np.isnan(data.Action))), 'Information'] = 0.0
data.loc[(np.isnan(data.Emotion)) & (~(np.isnan(data.Information) & np.isnan(data.Action))), 'Emotion'] = 0.0
data.loc[(np.isnan(data.Action)) & (~(np.isnan(data.Emotion) & np.isnan(data.Information))), 'Action'] = 0.0

# Убираю строки с NaN
data = data.dropna()
data = data.reset_index(drop=True)
data

pattern = "[^А-Яа-я-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

# Делю каждое сообщение на токены и привожу их к нормальной форме
def lemmatize(text):
    text = re.sub(pattern, ' ', text)
    tokens = []
    for token in text.split():
          token = token.strip('-') # убирает тире в начале предложения
          token = token.strip() # убирает пробелы в начале  
          token = morph.normal_forms(token)[0] # нормализуем   
          if (token != '') and (token not in stopwords_ru):                        
            tokens.append(token)   
    return tokens

lists_of_words = data['text'].apply(lemmatize)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(lists_of_words)

# Составляю список слов, которые используются(с их частотой)
word_index = tokenizer.word_index

# Обратно: по номеру получаю слово
reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key

# Кодирую словами цифрами
lists_of_numbers = tokenizer.texts_to_sequences(lists_of_words)
for msg in lists_of_numbers: 
  if(msg==[""]): 
    lists_of_numbers.remove("")

max = 0
for x in lists_of_numbers:
  if len(x) > max:
    max = len(x)

x = pad_sequences(lists_of_numbers_train, maxlen=max)
len(x)

y_inf = np.asarray(data['Information'])
y_emo = np.asarray(data['Emotion'])
y_act = np.asarray(data['Action'])

kfold = StratifiedKFold(n_splits=4, shuffle=True)

for train, test in kfold.split(x, y_inf):
  model_lstm_inf = Sequential()
  model_lstm_inf.add(Embedding(10000, 64, input_length=max))
  model_lstm_inf.add(Dropout(0.3))
  model_lstm_inf.add(LSTM(40, return_sequences=True))
  model_lstm_inf.add(Flatten())
  model_lstm_inf.add(Dense(1, activation='sigmoid'))

  model_lstm_inf.compile(optimizer='Adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', 'Recall', 'Precision'])
  
  model_lstm_inf_save_path = 'best_model_lstm_inf.h5'
  checkpoint_callback_lstm = ModelCheckpoint(model_lstm_inf_save_path, 
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
  
  history_lstm_inf = model_lstm_inf.fit(x[train], 
                              y_inf[train], 
                              epochs=4,
                              batch_size=32,
                              validation_split=0.4,
                              callbacks=[checkpoint_callback_lstm])

plt.plot(history_lstm_inf.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm_inf.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

for train, test in kfold.split(x, y_emo):
  model_lstm_emo = Sequential()
  model_lstm_emo.add(Embedding(10000, 64, input_length=max))
  model_lstm_emo.add(SpatialDropout1D(0.7))
  model_lstm_emo.add(LSTM(40, return_sequences=True))
  model_lstm_emo.add(Flatten())
  model_lstm_emo.add(Dense(1, activation='sigmoid'))

  model_lstm_emo.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', 'Recall', 'Precision'])
  
  model_lstm_emo_save_path= 'best_model_lstm_emo.h5'
  checkpoint_callback_lstm_emo = ModelCheckpoint(model_lstm_emo_save_path, 
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
  
  history_lstm_emo = model_lstm_emo.fit(x[train], 
                              y_emo[train], 
                              epochs=4,
                              batch_size=32,
                              validation_split=0.4,
                              callbacks=[checkpoint_callback_lstm])

plt.plot(history_lstm_emo.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm_emo.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

for train, test in kfold.split(x, y_act):
  model_lstm_act = Sequential()
  model_lstm_act.add(Embedding(10000, 64, input_length=max))
  model_lstm_act.add(SpatialDropout1D(0.6))
  model_lstm_act.add(LSTM(40, return_sequences=True))
  model_lstm_act.add(LSTM(40))
  model_lstm_act.add(Flatten())
  model_lstm_act.add(Dense(1, activation='sigmoid'))

  model_lstm_act.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', 'Recall', 'Precision'])
  
  model_lstm_save_path_act = 'best_model_lstm_act.h5'
  checkpoint_callback_lstm_act = ModelCheckpoint(model_lstm_save_path_act, 
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
  
  history_lstm_act = model_lstm_act.fit(x[train], 
                              y_act[train], 
                              epochs=5,
                              batch_size=32,
                              validation_split = 0.4,
                              callbacks=[checkpoint_callback_lstm_act])

plt.plot(history_lstm_act.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm_act.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()



