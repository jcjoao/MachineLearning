import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import ndimage
from keras.callbacks import EarlyStopping
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from keras.callbacks import TensorBoard
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import deeplake
import time

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D


##################################################################################################
# Modes:    0-MLP;    1-LogisticRegression;    2-Naive Bayes;    3-SVC;    4-Testing model       #
mode = 0
mode2 = 1
oversize = '1101'  # ON/OFF --- espelhos --- cores --- rotacoes
TEST_SIZE = 0.07
EPOCHS = 60
EARLY_STOPPING = True
PACIENCE = 5
##################################################################################################


x = np.load('Xtrain_Classification1.npy')
print(x.shape)
y = np.load('ytrain_Classification1.npy')
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

if oversize[0] == '1':
    class_1_indices = np.where(y_train == 1)[0]
    rotations = [90, 180, 270]
    augmented_x = []
    augmented_y = []
    for idx in class_1_indices:
        image = x_train[idx]  # Retrieve the image
        output = y_train[idx]  # Retrieve the label

        if oversize[1] == '1':
            # Add horizontal and vertical flips
            horizontal_flip = np.fliplr(image.reshape(28, 28, 3))
            vertical_flip = np.flipud(image.reshape(28, 28, 3))
            augmented_x.append(horizontal_flip.flatten())
            augmented_y.append(output)
            augmented_x.append(vertical_flip.flatten())
            augmented_y.append(output)

        if oversize[2] == '1':
            # Add darker and lighter images
            darker_image = np.clip(image * 0.9, 0, 255).astype(np.uint8)
            augmented_x.append(darker_image.flatten())
            augmented_y.append(output)
            lighter_image = np.clip(image * 1.1, 0, 255).astype(np.uint8)
            augmented_x.append(lighter_image.flatten())
            augmented_y.append(output)

        if oversize[3] == '1':
            # Add rotated images
            for angle in rotations:
                rotated_image = ndimage.rotate(image.reshape(28, 28, 3), angle,reshape=False)
                augmented_x.append(rotated_image.flatten())
                augmented_y.append(output)


    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)
    x_train = np.vstack((x_train, augmented_x))
    y_train = np.concatenate((y_train, augmented_y))
    print(len(x_train))
    print(len(x_test))


x_train = (x_train).astype('float32')/255.0
x_test = (x_test).astype('float32')/255.0


############################################################################
# MULTI LAYER PERCEPTRON
############################################################################
name = '/logs/'+str(time.time())
board = TensorBoard(log_dir=name,update_freq='epoch',profile_batch=0)

if mode==0:
    if mode2 == 0:
        model = Sequential()
        model.add(Dense(200, input_dim=2352, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(300, activation='relu'))   #THE GOOOAT
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        x_train = x_train.reshape(-1, 28, 28, 3)
        x_test = x_test.reshape(-1, 28, 28, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))


    adam = keras.optimizers.Adam(learning_rate = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=PACIENCE, verbose=1)

    if EARLY_STOPPING:
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=100, callbacks=[early_stopping,board], verbose=2)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=100, verbose=2)

    scores = model.evaluate(x_test, y_test)

    y_test_pred = model.predict(x_test)

    y_test_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_test_pred]
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred_binary)
    print(classification_report(y_test, y_test_pred_binary))

    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Balanced Accuracy: %.2f%%" % (balanced_acc*100))

    model.save('my_model.h5')
    

############################################################################
# Linear Models
############################################################################
if 0<mode<4:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

if mode==1:
    # logistic regression
    model = LogisticRegression(solver='sag', max_iter=1000)


if mode==2:
    # Gaussian Naive Bayes
    model = GaussianNB()


if mode==3:
    # SVM
    model = SVC(kernel='rbf', gamma='scale')

if mode==4:
    # QDA 
    X_train = x_train
    X_test = x_test
    model = QuadraticDiscriminantAnalysis(reg_param=0.9 , store_covariance=True)

if 0<mode<5:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)


#####################################################################################
#   TESTING
#####################################################################################
if mode ==5:

    model = keras.models.load_model('MLP90_30.h5')
    y_test_pred = model.predict(x_test)


if 0<mode<5:
    # Calculate accuracy
    y_test_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_test_pred]
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred_binary)
    accuracy = accuracy_score(y_test, y_test_pred_binary)
    print("Accuracy:", accuracy)
    print("Balanced Accuracy: %.2f%%" % (balanced_acc*100))

    print(classification_report(y_test, y_test_pred_binary))


#####################################################################################
#   Final
#####################################################################################

x_test_final = np.load('Xtest_Classification1.npy')
print(x_test_final.shape)
x_test_final = (x_test_final).astype('float32')/255.0

if mode == 0 and mode2 == 1:
    x_test_final = x_test_final.reshape(-1, 28, 28, 3)

y_test_final = model.predict(x_test_final)
y_test_final_binary = np.array([1.0 if pred >= 0.5 else 0.0 for pred in y_test_final])

print('resultadotesk3.npy: ' +  str(y_test_final_binary))
print(y_test_final_binary.shape)
print("How many 0s: ", np.count_nonzero(y_test_final_binary == 0))
print("How many 1s: ", np.count_nonzero(y_test_final_binary == 1))

np.save('resultadotesk3.npy', y_test_final_binary)
