# LSTM_감정분석(감성대화말뭉치)

## 1.  감성대화말뭉치 데이터
- AiHub - 감정분류 데이터
> ![Screen Shot 2021-12-28 at 21 29 49](https://user-images.githubusercontent.com/61719257/147566413-1e4f18da-5762-4746-bd3f-fa29314dafd1.png)
> ![Screen Shot 2021-12-28 at 21 30 10](https://user-images.githubusercontent.com/61719257/147566442-da893218-b76e-435a-b77c-5f681d109711.png)

## 2. 저번에는 tfidf + LSTM으로 성능이 아주 저조하게 나왔지만, 이번에는 word2vec embedding+LSTM 으로 좀 더 나은 성능을 보였음

## 3. model summary

![Screen Shot 2021-12-28 at 21 33 46](https://user-images.githubusercontent.com/61719257/147566695-6db989a0-73cc-4182-ab8a-41c0abdd4803.png)

## 4. evaluation

![Screen Shot 2021-12-28 at 21 34 22](https://user-images.githubusercontent.com/61719257/147566744-5280cc97-5614-4925-8fcf-d48a49b61577.png)
![Screen Shot 2021-12-28 at 21 34 38](https://user-images.githubusercontent.com/61719257/147566760-40dcb5b5-82cf-490c-afb2-06e89a0c2be5.png)
 
## 5. 결론

> 한국어 전처리에서 조사, 어미처리를 정교하게 해주지 않았기 때문에 큰 성능을 기대할 수는 없었지만 확실히 count-based representation 보다 distributed representation 이 더 나은 성능을 보입니다. 
> 임베딩 시 토크나이저의 num_words를 6000개(전체 말뭉치의 10%)로 설정하였는데 이 부분을 조금 늘려보면 성능향상이 되지 않을까라는 생각이 이제서야 듭니다. 지금 한번 바로 해봐야겠네요.


# ANN_감정분석

# 감성대화데이터 불러오기
```python
# Import libraies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import konlpy
import mecab

# read data
file_path = '감성대화/Training/감성대화말뭉치(원천데이터)_Training/감성대화말뭉치(원시데이터)_Training.xlsx'
df = pd.read_excel(file_path)
df.shape
## (74856, 14)

df.head()
```
![Screen Shot 2021-12-26 at 17 01 13](https://user-images.githubusercontent.com/61719257/147402376-93610f88-0b74-4f2a-8420-88235dfb1adf.png)
# Data 정제(생략) 후
```python
features = df.drop(['Emotion1', 'Emotion2'], axis=1)
targets = df[['Emotion1', 'Emotion2']] 

features.shape, targets.shape
## ((74855, 7), (74855, 2))
features.head()
targets.head()
```
![Screen Shot 2021-12-26 at 17 04 23](https://user-images.githubusercontent.com/61719257/147402422-8b183692-4a95-424f-b278-5288bf9b0d1b.png)
![Screen Shot 2021-12-26 at 17 04 34](https://user-images.githubusercontent.com/61719257/147402425-2161165d-ede0-43e3-b87b-49492ab23963.png)

# 한국어 불용어 사전 100개 불러오기
```python
with open('한국어불용어100.txt') as f:
    lines = f.readlines()
stopwords = []
for stopword in lines:
    stopwords.append(stopword.split('\t')[0])
print(stopwords)
```
```
['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '말하', '알', '그러나', '받', '못하', '일', '그런', '또', '문제', '더', '사회', '많', '그리고', '좋', '크', '따르', '중', '나오', '가지', '씨', '시키', '만들', '지금', '생각하', '그러', '속', '하나', '집', '살', '모르', '적', '월', '데', '자신', '안', '어떤', '내', '내', '경우', '명', '생각', '시간', '그녀', '다시', '이런', '앞', '보이', '번', '나', '다른', '어떻', '여자', '개', '전', '들', '사실', '이렇', '점', '싶', '말', '정도', '좀', '원', '잘', '통하', '소리', '놓']
```
# Tokenizing
```python
from category_encoders import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def word_preprocessing(word):
    remove_words = re.compile(r"[^ ㄱ-ㅣ가-힣a-zA-Z0-9+]")
    result = remove_words.sub('', word)
    return result

def tokenizer(raw, stopword=stopwords):
    
    m = mecab.MeCab()
    return [word for word in m.morphs(raw) if len(word) > 1 and word not in stopword ]

def data_prep(X):
    
    tfidf_tuned = TfidfVectorizer(stop_words=stopwords,
                              tokenizer=tokenizer,
                              ngram_range=(1,3),
                              min_df=2,
                              sublinear_tf=True
                              )
    
    X_1 = X[['Age', 'Gender', 'Situation', 'Condition']]
    X_2 = X[['Human1', 'Computer1', 'Computer2']]
    encoder = OrdinalEncoder()
    X_1_enc = pd.DataFrame(encoder.fit_transform(X), columns=['Age', 'Gender', 'Situation', 'Condition'])
    X_ = X_1_enc.copy()
    for col in ['Human1', 'Computer1', 'Computer2']:
        X_2[col] = X_2[col].apply(word_preprocessing)
        X_2_dtm_tfidf = tfidf_tuned.fit_transform(X_2[col])
        X_2_dtm_tfidf = pd.DataFrame(X_2_dtm_tfidf.todense(), columns=tfidf_tuned.get_feature_names_out())
        X_ = pd.concat((X_, X_2_dtm_tfidf), axis=1, join='inner')
        
    return X_

tfidf_tuned = TfidfVectorizer(stop_words=stopwords,
                              tokenizer=tokenizer,
                              ngram_range=(1,3),
                              min_df=3,
                              max_features=6000
                              )
features_ = features.copy()
features_['Human1'] = features_['Human1'].apply(word_preprocessing)
X_Human1_dtm_tfidf = tfidf_tuned.fit_transform(features_['Human1'])
X_Human1_dtm_tfidf = pd.DataFrame(X_Human1_dtm_tfidf.todense(), columns=tfidf_tuned.get_feature_names_out())
X_Human1_dtm_tfidf.shape
## (74855, 6000)

X_Human1_dtm_tfidf.head()
```
![Screen Shot 2021-12-26 at 17 07 03](https://user-images.githubusercontent.com/61719257/147402473-dc631c53-2a31-4637-aa11-0dc8f90eed4c.png)
# Data spliting
```python
from sklearn.model_selection import train_test_split

enc = OrdinalEncoder()
target = enc.fit_transform(targets['Emotion1'])
X_train, X_test, y_train, y_test = train_test_split(X_Human1_dtm_tfidf, target, 
                                                    test_size=0.15, stratify=target,
                                                    random_state=33)

X_train.shape, X_test.shape
##((63626, 6000), (11229, 6000))
y_train['Emotion1'] = y_train['Emotion1'].apply(lambda x: x-1)
y_test['Emotion1'] = y_test['Emotion1'].apply(lambda x: x-1)

y_train_oh = tf.one_hot(y_train['Emotion1'], 6)
y_test_oh = tf.one_hot(y_test['Emotion1'], 6)
y_train_oh
```
```
<tf.Tensor: shape=(63626, 6), dtype=float32, numpy=
array([[0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1.],
       ...,
       [0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 1.]], dtype=float32)>
```
# Build model : hyperparameters tuning - keras-tuner
- LSTM 모델은 성능이 좋지 않아서 기본 모델 사용
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Embedding, SimpleRNN, RNN, LSTMCell
from tensorflow.keras.models import Sequential
import IPython
import keras_tuner as kt
from tensorflow import keras

tf.random.set_seed(33)

def model_builder(hp):
    
    input_dim = 60
    output_size = 6
    hp_units = hp.Int('units', min_value = 32, max_value = 1012, step = 32)
    hp_units1 = hp.Int('units1', min_value = 32, max_value = 1012, step = 32)
    hp_units2 = hp.Int('units2', min_value = 32, max_value = 712, step = 32)
    hp_units3 = hp.Int('units3', min_value = 32, max_value = 512, step = 32)
    # hp_units4 = hp.Int('units4', min_value = 32, max_value = 512, step = 2)
    # hp_units5 = hp.Int('units5', min_value = 32, max_value = 512, step = 2)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
    hp_kernel_init = hp.Choice('kernel_initializer', values = ['he_normal', 'he_uniform']) 
    
    # The LSTM layer with default options uses CuDNN.
    # lstm_layer = keras.layers.LSTM(hp_units, 
    #                                activation='tanh', 
    #                                recurrent_activation='sigmoid',
    #                                input_shape=(None, input_dim))
    
    model = keras.models.Sequential(
        [
            # lstm_layer,
            keras.layers.Dense(hp_units, activation='relu', 
                               kernel_initializer=hp_kernel_init, 
                               input_shape=(6000,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hp_units1, kernel_initializer=hp_kernel_init, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hp_units1, kernel_initializer=hp_kernel_init, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hp_units2, kernel_initializer=hp_kernel_init, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hp_units3, kernel_initializer=hp_kernel_init, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size, activation='softmax'),
        ]
    )
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 100,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'kt5_LSTM_result')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                           min_delta=0, 
                                           patience=10, 
                                           verbose=1)

checkpoint_filepath = "best.hdf5"
save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True,
    save_weights_only=True, 
    mode='auto', 
    save_freq='epoch', 
    options=None)

tuner.search(X_train, y_train_oh, 
             epochs=100, 
             batch_size=5000, 
             callbacks=[ClearTrainingOutput(), early_stop, save_best], 
             validation_split=0.15)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

model = tuner.hypermodel.build(best_hps)

model.summary()
```
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_6 (Dense)             (None, 992)               5952992   
                                                                 
 batch_normalization_5 (Batc  (None, 992)              3968      
 hNormalization)                                                 
                                                                 
 dense_7 (Dense)             (None, 992)               985056    
                                                                 
 batch_normalization_6 (Batc  (None, 992)              3968      
 hNormalization)                                                 
                                                                 
 dense_8 (Dense)             (None, 992)               985056    
                                                                 
 batch_normalization_7 (Batc  (None, 992)              3968      
 hNormalization)                                                 
                                                                 
 dense_9 (Dense)             (None, 128)               127104    
                                                                 
 batch_normalization_8 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 dense_10 (Dense)            (None, 64)                8256      
                                                                 
 batch_normalization_9 (Batc  (None, 64)               256       
 hNormalization)                                                 
                                                                 
 dense_11 (Dense)            (None, 6)                 390       
                                                                 
=================================================================
Total params: 8,071,526
Trainable params: 8,065,190
Non-trainable params: 6,336
_________________________________________________________________
```
# Model fit with the best model
```python
model.fit(X_train, y_train_oh, 
          epochs=300,
          batch_size=5000,
          callbacks=[early_stop, save_best],
          validation_split=0.15)

model.load_weights(checkpoint_filepath)
test_loss, test_acc = model.evaluate(X_test,  y_test_oh, verbose=1)
```
```
351/351 [==============================] - 4s 11ms/step - loss: 1.4165 - accuracy: 0.4462
```
# Evaluation
```python
from sklearn.metrics import classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))
```
```
              precision    recall  f1-score   support

           0       0.83      0.52      0.64      1867
           1       0.41      0.64      0.50      1858
           2       0.29      0.76      0.42      1906
           3       0.62      0.24      0.35      1696
           4       0.71      0.23      0.35      2026
           5       0.67      0.28      0.39      1876

    accuracy                           0.45     11229
   macro avg       0.59      0.45      0.44     11229
weighted avg       0.59      0.45      0.44     11229
```


