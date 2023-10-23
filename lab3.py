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

import deeplake

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


##################################################################################################
# Modes:    0-MLP;    1-LogisticRegression;    2-Naive Bayes;    3-SVC;    4-Testing model       #
mode = 0
mode2 = 1
TEST_SIZE = 0.10
EPOCHS = 400
EARLY_STOPPING = True
##################################################################################################


x = np.load('Xtrain_Classification1.npy')
print(x.shape)
y = np.load('ytrain_Classification1.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

class_1_indices = np.where(y_train == 1)[0]
rotations = [90, 180, 270]
augmented_x = []
augmented_y = []
for idx in class_1_indices:
    image = x_train[idx]  # Retrieve the image
    output = y_train[idx]  # Retrieve the label

    # Add horizontal and vertical flips
    horizontal_flip = np.fliplr(image.reshape(28, 28, 3))
    vertical_flip = np.flipud(image.reshape(28, 28, 3))
    augmented_x.append(horizontal_flip.flatten())
    augmented_y.append(output)
    augmented_x.append(vertical_flip.flatten())
    augmented_y.append(output)

    # #Add darker and lighter images
    # darker_image = np.clip(image * 0.9, 0, 255).astype(np.uint8)
    # augmented_x.append(darker_image.flatten())
    # augmented_y.append(output)
    # lighter_image = np.clip(image * 1.1, 0, 255).astype(np.uint8)
    # augmented_x.append(lighter_image.flatten())
    # augmented_y.append(output)

    # Add rotated images
    for angle in rotations:
        rotated_image = ndimage.rotate(image.reshape(28, 28, 3), angle,reshape=False)
        augmented_x.append(rotated_image.flatten())
        augmented_y.append(output)


augmented_x = np.array(augmented_x)
augmented_y = np.array(augmented_y)
x_train = np.vstack((x_train, augmented_x))
y_train = np.concatenate((y_train, augmented_y))
print("AAAAAAAAAA")
# print(np.count_nonzero (y_train == 1))
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
x_train = (x_train).astype('float32')/255.0
x_test = (x_test).astype('float32')/255.0
#train_labels = keras.utils.to_categorical(y,2)


############################################################################
# MULTI LAYER PERCEPTRON
############################################################################

if mode==0:
    if mode2 == 0:
        model = Sequential()
        model.add(Dense(200, input_dim=2352, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(250, activation='relu'))   #THE GOOOAT
        model.add(Dense(50, activation='relu'))
        #model.add(Dense(25, activation='relu'))
        #output is either 0 or 1, so we use sigmoid
        model.add(Dense(1, activation='sigmoid'))
    else:
        x_train = x_train.reshape(-1, 28, 28, 3)
        x_test = x_test.reshape(-1, 28, 28, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))  # Adjust input shape as needed
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #dense 128
        model.add(Flatten())
        # model.add(Dense(200, activation='relu'))
        # model.add(Dense(250, activation='relu'))
        # model.add(Dense(250, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(learning_rate = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=12, verbose=1)

    if EARLY_STOPPING:
        model.fit(x_train, y_train, epochs=400, batch_size=100, callbacks=[early_stopping], verbose=2)
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
    
    exit()




############################################################################
# Linear Models
############################################################################

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
# Normalize features
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

if mode==1:
    # Initialize logistic regression model
    model = LogisticRegression(solver='sag', max_iter=10000)


if mode==2:
    # Initialize Gaussian Naive Bayes model
    model = GaussianNB()


if mode==3:
    # Initialize SVM model (using a radial basis function kernel)
    model = SVC(kernel='rbf', gamma='scale')

if 0<mode<4:
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_test_pred = model.predict(X_test)


#####################################################################################
#   TESTING
#####################################################################################
if mode ==4:

    model = keras.models.load_model('MLP90_30.h5')
    y_test_pred = model.predict(x_test)



# Calculate accuracy
y_test_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_test_pred]
balanced_acc = balanced_accuracy_score(y_test, y_test_pred_binary)
accuracy = accuracy_score(y_test, y_test_pred_binary)
print("Accuracy:", accuracy)
print("Balanced Accuracy: %.2f%%" % (balanced_acc*100))