import numpy as np
import pandas as pd
import os
import re
import nltk
nltk.download('punkt')
import tensorflow
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup

from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential

# read in the train and test tsv files
train= pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")

# clear all noise info from phrases
def preprocessing(df):
    results = []
    for phrase in df['Phrase']:
        # get only the plain text content
        phrase_text = BeautifulSoup(phrase,"html.parser").get_text()
        phrase_text.replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", " ")
        tokens = word_tokenize(phrase_text)
        lemma_words = []
        for token in tokens:
            lemma_words.append(lemmatizer.lemmatize(token))
        results.append(lemma_words)
    return(results)

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

train_reviews = preprocessing(train)
test_reviews = preprocessing(test)

target=train.Sentiment.values
y_target=to_categorical(target)

X_train,X_val,y_train,y_val=train_test_split(train_reviews,y_target,test_size=0.1,stratify=y_target)

len_max = find_max_list(X_train)

vocabulary = set()
vocab_length = 0

for phrase in X_train:
    vocabulary.update(phrase)
vocab_length = len(list(vocabulary))

temp = X_train
def texts_to_sequences(texts):
    tokenizer = Tokenizer(num_words=vocab_length)
    tokenizer.fit_on_texts(list(temp))
    texts = tokenizer.texts_to_sequences(texts)
    texts = sequence.pad_sequences(texts, maxlen=len_max)
    return texts

X_train = texts_to_sequences(X_train)
X_val = texts_to_sequences(X_val)

tokenizer = Tokenizer(num_words=vocab_length)
tokenizer.fit_on_texts(list(temp))
X_test = texts_to_sequences(test_reviews)
X_test = sequence.pad_sequences(X_test, maxlen = len_max)

print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)

model=Sequential()
model.add(Embedding(vocab_length,100,input_length=len_max))
model.add(LSTM(64,dropout=0.5,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.summary()

history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=6, batch_size=256, verbose=1)

epoch_count = range(1, len(history.history['loss']) + 1)

plt.plot(epoch_count, history.history['loss'], 'r--')
plt.plot(epoch_count, history.history['val_loss'], 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=-1)

sub_file = pd.read_csv('sampleSubmission.csv',sep=',')
sub_file.Sentiment=y_pred
sub_file.to_csv('Submission.csv',index=False)