import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization,Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

x_train = np.load('./models_data/title_x_train_wordsize9983.npy', allow_pickle=True)
x_test = np.load('./models_data/title_x_test_wordsize9983.npy', allow_pickle=True)
y_train = np.load('./models_data/title_y_train_wordsize9983.npy', allow_pickle=True)
y_test = np.load('./models_data/title_y_test_wordsize9983.npy', allow_pickle=True)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(9983, 100,trainable=True,input_length=100))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True,
                             kernel_regularizer=l2(0.001),recurrent_dropout=0.2)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True,
                          kernel_regularizer=l2(0.001), dropout=0.2)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=False,
                          kernel_regularizer=l2(0.001), dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))



adam=tf.keras.optimizers.Adam(learning_rate=0.001 ,beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])
model.summary()

fit_hist = model.fit(x_train, y_train, batch_size=64,
                     epochs=30, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy', score[1])
model.save('./models/news_section_classfication_model_{}.h5'.format(score[1]))
# plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
# plt.plot(fit_hist.history['accuracy'], label='accuracy')
# plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(fit_hist.history['accuracy'], label='Train Accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(fit_hist.history['loss'], label='Train Loss')
plt.plot(fit_hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()