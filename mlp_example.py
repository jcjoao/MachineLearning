# -*- coding: utf-8 -*-
"""MLP_example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H7sjXaB8HwmKq2WvWW62ULyj7hM7SGym
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

batch_size=500
epochs=150
lr=0.001

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks/

X=np.load('Xtrain1.npy')
y=np.load('ytrain1.npy')


train_images = (X).astype('float32')/255.0
train_labels = keras.utils.to_categorical(y,2)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2)

model_MLP = Sequential()
model_MLP.add(Dense(16,activation = 'relu',input_dim = 2700))
model_MLP.add(Dense(8,activation = 'relu'))
model_MLP.add(Dense(2,activation = 'softmax'))

model_MLP.summary()

adam = keras.optimizers.Adam(learning_rate = lr)
model_MLP.compile(optimizer = adam,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])



#MLP without early stopping
history = model_MLP.fit(x = X_train,y=y_train,epochs = epochs,batch_size=batch_size,validation_data = (X_val,y_val),verbose = 1)


####PLOT EVOLUTION
plt.figure(1)
plt.clf()
plt.plot(history.history['loss'], label='train'),
plt.plot(history.history['val_loss'], label='train'), plt.show()



X=np.load('Xtest1.npy')
test_images = (X).astype('float32')/255.0
results_MLP = np.argmax(model_MLP.predict(test_images),1)