import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

def rounder(array, thresh=.5):
    new = []
    for i in array:
        if i >= thresh:
            new.append(1)
        else:
            new.append(0)
    return np.array(new)

sarcasm = pd.read_json('../data/raw/Sarcasm_Headlines_Dataset.json', lines=True)
sarcasm.columns = ['article_link', 'headline', 'is_parody']
sarcasm.loc[sarcasm['article_link'].str.contains('comhttp'), 'article_link'] = sarcasm.loc[sarcasm['article_link'].str.contains('comhttp'), 'article_link'].str[30:]
sarcasm.iloc[19948, 2] = 1

X_train, X_test, y_train, y_test = train_test_split(sarcasm['headline'], y, test_size=.20)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(X_train))
list_tokenized_train = tokenizer.texts_to_sequences(X_train)
list_tokenized_test = tokenizer.texts_to_sequences(X_test)

total_word_counts = [len(headline) for headline in list_tokenized_train]

maxlen = 18
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

tokens = tokenizer.sequences_to_texts(list_tokenized_train) + tokenizer.sequences_to_texts(list_tokenized_test)
tokens = [t.split(' ') for t in tokens]

model = Word2Vec(tokens, size=100, window=5, min_count=1, workers=4)
model.train(tokens, total_examples=model.corpus_count, epochs=10)

embeddings = model.wv
keras_weights = embeddings.get_keras_embedding()

model = Sequential()
model.add(keras_weights)
model.add(Conv1D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling1D())
model.add(Conv1D(64, 3, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

batch_size = 256
num_epochs = 30

hist = model.fit(X_t, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)

y_pred = model.predict(X_te)
y_pred_bin = rounder(y_pred)
accuracy_score(y_test, y_pred_bin)