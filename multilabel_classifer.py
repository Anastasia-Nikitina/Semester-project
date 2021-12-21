# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# %tensorflow_version 2.x

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Dropout, LSTM, Bidirectional, SpatialDropout1D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re, nltk
import keras.utils
from pymorphy2 import MorphAnalyzer
nltk.download('stopwords')

# %matplotlib inline


data = pd.read_excel('data_ver00.xlsx')

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

text = data['text']

pattern = "[^А-Яа-я-]+"
pattern2 = "^[А-Яа-я]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()
print(stopwords_ru)

# Делю каждое сообщение на токены и привожу их к нормальной форме
def lemmatize(text):
    text = re.sub(pattern, ' ', text)
    tokens = []
    for token in text.split():
      if token not in stopwords_ru:
          token = token.strip('-') # убирает тире в начале предложения
          token = token.strip() # убирает пробелы в начале   
          if (token != ''):    
              token = morph.normal_forms(token)[0]            
              tokens.append(token)
    return tokens


lists_of_words = text.apply(lemmatize)
lists_of_words

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(lists_of_words)

# Составляю список слов, которые используются(с их частотой)
word_index = tokenizer.word_index
word_index

# Обратно: по номеру получаю слово
reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key

# Проверка: вывожу 20 самых частоиспользуемых слов

for i in range(1, 21):
    print(i, ':', reverse_word_index[i])

# Кодирую словами цифрами
lists_of_numbers = tokenizer.texts_to_sequences(lists_of_words)
for msg in lists_of_numbers: 
  if(msg==[""]): 
    lists_of_numbers.remove("")

max = 0
for x in lists_of_numbers:
  if len(x) > max:
    max = len(x)

x_train = (pad_sequences(lists_of_numbers, maxlen=max))

# Создаю массив с правильными ответами
y_train = (np.asarray(data[['Information', 'Emotion', 'Action']])).astype('float32').reshape((-1,3))
y_train

model_lstm = Sequential()
model_lstm.add(Embedding(10000, 128, input_length=1500))
model_lstm.add(SpatialDropout1D(0.5))
model_lstm.add(LSTM(40, return_sequences=True))
model_lstm.add(LSTM(40))
model_lstm.add(Dense(3, activation='sigmoid'))

model_lstm.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model_lstm_save_path = 'best_model_lstm.h5'
checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

history_lstm = model_lstm.fit(x_train, 
                              y_train, 
                              epochs=5,
                              batch_size=32,
                              validation_split=0.4,
                              callbacks=[checkpoint_callback_lstm])

plt.plot(history_lstm.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()