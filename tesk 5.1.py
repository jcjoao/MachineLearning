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

x = np.load('Xtrain_Classification1.npy')
print(x)
print(x.shape)
y = np.load('ytrain_Classification1.npy')

class_1_indices = np.where(y == 1)[0]
rotations = [90, 180, 270]
augmented_x = []
augmented_y = []

for idx in class_1_indices:
    image = x[idx]  # Retrieve the image
    output = y[idx]  # Retrieve the label
    for angle in rotations:
        rotated_image = ndimage.rotate(image.reshape(28, 28, 3), angle,reshape=False)
        augmented_x.append(rotated_image.flatten())
        augmented_y.append(output)

augmented_x = np.array(augmented_x)
augmented_y = np.array(augmented_y)
x = np.vstack((x, augmented_x))
y = np.concatenate((y, augmented_y))

train_images = (x).astype('float32')/255.0
#train_labels = keras.utils.to_categorical(y,2)
x_train, x_test, y_train, y_test = train_test_split(train_images, y, test_size=0.2, random_state=42)

mode = 0

if mode == 0:
    model = Sequential()
    model.add(Dense(100, input_dim=2352, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(25, activation='relu'))
    #output is either 0 or 1, so we use sigmoid
    model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(learning_rate = 0.001)
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=9, verbose=1)
    
    model.fit(x_train, y_train, epochs=400, batch_size=100, callbacks=[early_stopping])
    
    
    scores = model.evaluate(x_test, y_test)

    y_test_pred = model.predict(x_test)

    y_test_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_test_pred]
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred_binary)

    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Balanced Accuracy: %.2f%%" % (balanced_acc*100))

    print(y_test[:10])
    #plot the first 10 images
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i].reshape(28,28,3))
        #title is the prediction, either 0 or 1
        plt.title(f"{int(y_test[i])}, {int(np.round(y_test_pred[i]))}")
        plt.axis('off')
    plt.show()

if mode == 1:
    clf = MLPClassifier(hidden_layer_sizes=(200), max_iter=100, alpha=0.0001, solver='adam', random_state=42, activation = "relu", early_stopping=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100))
    print(classification_report(y_test, y_pred))
    #plot the first 10 images
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i].reshape(28,28,3))
        #title is the prediction, either 0 or 1
        plt.title(np.round(y_pred[i]))
        plt.axis('off')
    plt.show()




