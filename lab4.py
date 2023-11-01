import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
import tensorflow
from PIL import Image
from keras.optimizers import Adam

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from keras.callbacks import TensorBoard
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.utils import to_categorical

import time
import random

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import Callback
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications import ResNet50


class BalancedAccuracyCallback(Callback):
    def __init__(self, validation_data, numclasses):
        super(BalancedAccuracyCallback, self).__init__()
        self.validation_data = validation_data
        self.numclasses = numclasses

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_test_pred = self.model.predict(x_val)

        if self.numclasses == 2:
            y_test_pred = [1 if pred > 0.5 else 0 for pred in y_test_pred]
        else:
            y_test_pred = [np.argmax(pred) for pred in y_test_pred]
            y_val = [np.argmax(pred) for pred in y_val]

        balanced_acc = balanced_accuracy_score(y_val , y_test_pred)

        print(f'Balanced Accuracy: {balanced_acc:.4f}')
        if balanced_acc > 2:
            print('PARAAAAAAAAAAAAAAA')
            self.model.stop_training = True


def oversample(label,oversize):
    class_1_indices = np.where(y_train == label)[0]
    rotations = [90, 180, 270]
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

############################################################################
# MULTI LAYER PERCEPTRON
############################################################################

def MLP(numclasses):
    model = Sequential()
    model.add(Dense(200, input_dim=2352, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(300, activation='relu'))   #THE GOOOAT
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    if numclasses == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(numclasses, activation='softmax'))
    return model

def CNN(numclasses):
    # x_train = x_train.reshape(-1, 28, 28, 3)
    # x_test = x_test.reshape(-1, 28, 28, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu')) #comentada
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    if numclasses == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(numclasses, activation='softmax'))
    return model

def CNN2(numclasses):
    # x_train = x_train.reshape(-1, 28, 28, 3)
    # x_test = x_test.reshape(-1, 28, 28, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if numclasses == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(numclasses, activation='softmax'))
    return model

def neuralnetcompiler(model, x_train, y_train, x_test, y_test, numclasses):
    global modelarray
    if numclasses != 2:
        y_train = to_categorical(y_train, num_classes=numclasses)
        y_test = to_categorical(y_test, num_classes=numclasses)

    adam = keras.optimizers.Adam(learning_rate = 0.001)

    if numclasses == 2:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=PACIENCE, verbose=1)
    balanced_accuracy_callback = BalancedAccuracyCallback(validation_data=(x_test, y_test),numclasses=numclasses)


    iter = len(modelarray)
    if EARLY_STOPPING:
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS[iter], batch_size=100, callbacks=[early_stopping,board,balanced_accuracy_callback], verbose=2)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS[iter], batch_size=100, verbose=2)

    modelarray.append(model)

    y_test_pred = model.predict(x_test)

    if numclasses == 2:
        y_test_pred = [1 if pred > 0.5 else 0 for pred in y_test_pred]
    else:
        y_test_pred = [np.argmax(pred) for pred in y_test_pred]
    return y_test_pred

def accuracyscore(y_test, y_test_pred):
    # Calculate accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)
    print("Balanced Accuracy: %.2f%%" % (balanced_acc*100))
    print(classification_report(y_test, y_test_pred))
    # model.save('my_modelo.h5')
    
############################################################################
# Linear Models
############################################################################
    
def logisticregression(x_train, y_train, x_test):
    # logistic regression
    scaler = StandardScaler()
    model = LogisticRegression(solver='sag', max_iter=1000)
    y_test_pred = linearcompile(model, scaler.fit_transform(x_train), y_train, scaler.transform(x_test))
    return y_test_pred

def gaussianNB(x_train, y_train, x_test):
    scaler = StandardScaler()
    # Gaussian Naive Bayes
    model = GaussianNB()
    y_test_pred = linearcompile(model, scaler.fit_transform(x_train), y_train, scaler.transform(x_test))
    return y_test_pred
    
def svc(x_train, y_train, x_test):
    scaler = StandardScaler()
    # SVM
    model = SVC(kernel='rbf', gamma='scale')
    y_test_pred = linearcompile(model, scaler.fit_transform(x_train), y_train, scaler.transform(x_test))
    return y_test_pred

def QDA(x_train, y_train, x_test):
    QDA 
    model = QuadraticDiscriminantAnalysis(reg_param=0.9 , store_covariance=True)
    y_test_pred = linearcompile(model, x_train, y_train, x_test)
    return y_test_pred

def linearcompile(model, x_train, y_train, x_test):
    global modelarray
    
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    
    modelarray.append(model)
    return y_test_pred

############################################################################
# Mobile Net
############################################################################

def mobilenet(x_train, y_train, x_test, numclasses):
    x_train = x_train.reshape(-1, 28, 28, 3)
    x_test = x_test.reshape(-1, 28, 28, 3)
    if numclasses != 2:
        y_train = to_categorical(y_train, num_classes=numclasses)
    
    x_train0 = []


    padding = ((2, 2), (2, 2), (0, 0))  # Add 2 pixels of padding to each side, only along height and width
    for img in x_train:
        # Pad the image with white pixels (represented as 255 in uint8 format)
        padded_img = np.pad(img, padding, mode='constant', constant_values=255)

        # Append resized image to the list
        x_train0.append(padded_img)
    x_train = np.array(x_train0)
    
    # Load the ResNet50 model with pre-trained weights and without the top classification layers
    base_model = ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(3, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)

    y_test_pred = model.predict(x_test)
    y_test_pred = [np.argmax(pred) for pred in y_test_pred]
    
    return y_test_pred
        
def choosemodelmode(modelmode, x_train, y_train, x_test, y_test, numclasses):
    if modelmode == 0:
        model = MLP(numclasses)
        y_test_pred = neuralnetcompiler(model, x_train, y_train, x_test, y_test, numclasses)

    if modelmode == 1:
        model = CNN(numclasses)
        y_test_pred = neuralnetcompiler(model, x_train.reshape(-1, 28, 28, 3), y_train, x_test.reshape(-1, 28, 28, 3), y_test, numclasses)

    if modelmode == 2:
        y_test_pred = logisticregression(x_train, y_train, x_test)

    if modelmode == 3:
        y_test_pred = gaussianNB(x_train, y_train, x_test)

    if modelmode == 4:
        y_test_pred = svc(x_train, y_train, x_test)

    if modelmode == 5:
        y_test_pred = QDA(x_train, y_train, x_test)
    
    if modelmode == 6:
        y_test_pred = mobilenet(x_train, y_train, x_test, numclasses)

    if modelmode == 7:
        model = CNN2(numclasses)
        y_test_pred = neuralnetcompiler(model, x_train.reshape(-1, 28, 28, 3), y_train, x_test.reshape(-1, 28, 28, 3), numclasses)

    return y_test_pred


##################################################################################################
# CONTROL PANEL
##################################################################################################
# Modes:    0 - Only one model;  1- 3 model method;     -1 -Load Model
mode = 1
modelmode = [4,4,4] # 0-MLP;  1-CNN;  2-LogisticRegression;    3-Naive Bayes;    4-SVC;   5-QDA;  -1 -Load Model; 6-Mobilenet; 7- CNN2
oversize = '2101'  # OFF/Auto/Manual --- espelhos +2 --- cores +2 --- rotacoes +3
TEST_SIZE = 0.10
PACIENCE = 5
EPOCHS = [1,1,50]
EARLY_STOPPING = True
downsample = 0.5
##################################################################################################


x = np.load('Xtrain_Classification2.npy')
print(x.shape)
y = np.load('ytrain_Classification2.npy')
print(y.shape)


labels = 'Nevu 0', ' Melanoma 1', 'Vascular Lesions 2', 'Granulocytes 3', 'Basophils 4', 'Lymphocytes 5'

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

sizes = [len(np.where(y_train==i)[0]) for i in range(6)]
print(sizes)
# plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# plt.show()

if oversize[0] == '1':
    #Automatic Balanced Oversampling
    indices = np.where(y_train==0)[0]
    indices = np.array(random.sample(indices.tolist(), int(sizes[0]*downsample)))
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)
    sizes = [len(np.where(y_train==i)[0]) for i in range(6)]
    print(sizes)
    
    for i in range(1,6):
        augmented_x = []
        augmented_y = []
        oversample(i,oversize)

        indices = [i for i in range(len(augmented_x))]
        new_size = min(int((sizes[0]/sizes[i]-1)*sizes[i]), len(augmented_x))
        indices = np.array(random.sample(indices, new_size))

        augmented_x = np.array(augmented_x)
        augmented_x = augmented_x[indices]
        augmented_y = np.array(augmented_y)
        augmented_y = augmented_y[indices]
        x_train = np.vstack((x_train, augmented_x))
        y_train = np.concatenate((y_train, augmented_y))
    
elif oversize[0] == '2':
    #[4531, 769, 97, 1981, 848, 808]
    #Manual Balanced Oversampling
    augmented_x = []
    augmented_y = []
    # downsample(x_train, y_train, 0,downsamplepercentage)
    if downsample != 0:
        indices = np.where(y_train==0)[0]
        indices = np.array(random.sample(indices.tolist(), int(sizes[0]*downsample)))
        x_train = np.delete(x_train, indices, axis=0)
        y_train = np.delete(y_train, indices, axis=0)
    #oversample(0,'0000')
    #[4800, 813, 102, 2082, 899, 870]
    #[2400, 2439, 816, 2082, 2697, 2610]
    oversample(1,'0101')
    oversample(2,'0111')
    oversample(3,'0000')
    oversample(4,'0100')
    oversample(5,'0110')

    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)
    x_train = np.vstack((x_train, augmented_x))
    y_train = np.concatenate((y_train, augmented_y))
    print(len(x_train))
    print(len(x_test))

balanced_sizes = [len(np.where(y_train==i)[0]) for i in range(6)]
print(balanced_sizes)
# plt.pie(balanced_sizes, labels=labels, autopct='%1.1f%%')
# plt.show()

#Normalize the Data
x_train = (x_train).astype('float32')/255.0
x_test = (x_test).astype('float32')/255.0

name = '/logs/'+str(time.time())
board = TensorBoard(log_dir=name,update_freq='epoch',profile_batch=0)
modelarray = []

#####################################################################################
# Only one model for all the data
#####################################################################################
if mode == 0:
    y_test_pred = choosemodelmode(modelmode[0], x_train, y_train, x_test, y_test, 6)
    accuracyscore(y_test, y_test_pred)

#####################################################################################
# Separate the data in 2 parts, and apply a different model to each part
#####################################################################################
if mode == 1:
    y_train0 = np.array([0 if category < 3 else 1 for category in y_train])
    y_test0 = np.array([0 if category < 3 else 1 for category in y_test])
    y_test_pred0 = choosemodelmode(modelmode[0], x_train, y_train0, x_test, y_test0, 2)
    
    print("Accuracy Score for the First Division:")
    accuracyscore(y_test0, y_test_pred0)
    
    print(len(y_test_pred0))
    
    # Separate the data
    x_train1 = np.array([x_train[i] for i in range(len(y_train0)) if y_train0[i] == 0])
    y_train1 = np.array([y_train[i] for i in range(len(y_train0)) if y_train0[i] == 0])

    x_test1 = np.array([x_test[i] for i in range(len(y_test0)) if y_test0[i] == 0])
    y_test1 = np.array([y_test[i] for i in range(len(y_test0)) if y_test0[i] == 0])
    
    #tururu
    #y_train1 = np.array([0 if category < 2 else 1 for category in y_train1])
    #y_test1 = np.array([0 if category < 2 else 1 for category in y_test1])

    x_train2 = np.array([x_train[i] for i in range(len(y_train0)) if y_train0[i] == 1])
    y_train2 = np.array([y_train[i] for i in range(len(y_train0)) if y_train0[i] == 1])

    x_test2 = np.array([x_test[i] for i in range(len(y_test0)) if y_test0[i] == 1])
    y_test2 = np.array([y_test[i] for i in range(len(y_test0)) if y_test0[i] == 1])

    print(x_train1.shape)
    print(x_train2.shape)

    y_test2 = [cat-3 for cat in y_test2]
    y_train2 = [cat-3 for cat in y_train2]

    #Apply the models
    y_test_pred1 = choosemodelmode(modelmode[1], x_train1, y_train1, x_test1, y_test1, 3)
    y_test_pred2 = choosemodelmode(modelmode[2], x_train2, y_train2, x_test2, y_test2, 3)

    y_train2 = [cat+3 for cat in y_train2]
    y_test2 = [cat+3 for cat in y_test2]
    y_test_pred2 = [cat+3 for cat in y_test_pred2]

    print("Accuracy Score for the First Half:")
    accuracyscore(y_test1, y_test_pred1)
    print("Accuracy Score for the Second Half:")
    accuracyscore(y_test2, y_test_pred2)

    #Final Balanced Accuracy
    y_test_pred_final = np.concatenate((y_test_pred1, y_test_pred2))
    y_test_final = np.concatenate((y_test1, y_test2))
    print("Accuracy Score for the Final Result:")
    accuracyscore(y_test_final, y_test_pred_final)

#####################################################################################
#   TESTING WITH A LOADED MODEL
#####################################################################################
if mode == -1:
    model = keras.models.load_model('MLP90_30.h5')
    y_test_pred = model.predict(x_test)


#####################################################################################
#   Final
#####################################################################################

x_test_final = np.load('Xtest_Classification2.npy')
print(x_test_final.shape)
x_test_final = (x_test_final).astype('float32')/255.0

if mode == 0 or mode == -1:
    if modelmode[0] == 1 or modelmode[0] == 7:
        x_test_final = x_test_final.reshape(-1, 28, 28, 3)
    y_test_final = modelarray[0].predict(x_test_final)
    y_test_pred = [np.argmax(pred) for pred in y_test_pred]

if mode == 1:
    y_test_final0 = modelarray[0].predict(x_test_final.reshape(-1, 28, 28, 3))
    modelarray[0].save('my_modelo0.h5')
    modelarray[1].save('my_modelo1.h5')
    modelarray[2].save('my_modelo2.h5')


    y_test_final0 = [1 if pred > 0.5 else 0 for pred in y_test_final0]

    x_test_final1 = np.array([x_test_final[i] for i in range(len(y_test_final0)) if y_test_final0[i] == 0])
    x_test_final2 = np.array([x_test_final[i] for i in range(len(y_test_final0)) if y_test_final0[i] == 1])

    if modelmode[1]== 1 or modelmode[1] == 7:
        x_test_final1 = x_test_final1.reshape(-1, 28, 28, 3)
    if modelmode[2]== 1 or modelmode[2] == 7:
        x_test_final2 = x_test_final2.reshape(-1, 28, 28, 3)
        
    #Predict
    y_test_final1 = modelarray[1].predict(x_test_final1)
    y_test_final2 = modelarray[2].predict(x_test_final2)

    #Convert into the correct form
    y_test_final1 = [np.argmax(pred) for pred in y_test_final1]
    y_test_final2 = [np.argmax(pred) for pred in y_test_final2]

    y_test_final2 = [cat+3 for cat in y_test_final2]

    indices1 = [i for i in range(len(y_test_final0)) if y_test_final0[i] == 0]
    indices2 = [i for i in range(len(y_test_final0)) if y_test_final0[i] == 1]
    y_test_final = np.zeros(len(y_test_final0))
    y_test_final[indices1] = y_test_final1
    y_test_final[indices2] = y_test_final2


print('resultadotesk4.npy: ' +  str(y_test_final))
print(y_test_final.shape)
print("How many 0s: ", np.count_nonzero(y_test_final == 0))
print("How many 1s: ", np.count_nonzero(y_test_final == 1))
print("How many 2s: ", np.count_nonzero(y_test_final == 2))
print("How many 3s: ", np.count_nonzero(y_test_final == 3))
print("How many 4s: ", np.count_nonzero(y_test_final == 4))
print("How many 5s: ", np.count_nonzero(y_test_final == 5))

np.save('resultadotesk4.npy', y_test_final)


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
