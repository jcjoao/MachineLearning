import os

from sklearn.metrics import balanced_accuracy_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import keras



x = np.load('Xtrain_Classification2.npy')
print(x.shape)
y = np.load('ytrain_Classification2.npy')
print(y.shape)

modelarray = [keras.models.load_model('my_modelo0.h5'), keras.models.load_model('bom1-82.h5'), keras.models.load_model('bom2-94.h5')]


x_test_final = np.load('Xtest_Classification2.npy')
print(x_test_final.shape)
x_test_final = (x_test_final).astype('float32')/255.0


y_test_final0 = modelarray[0].predict(x_test_final.reshape(-1, 28, 28, 3))

y_test_final0 = [1 if pred > 0.5 else 0 for pred in y_test_final0]

x_test_final1 = np.array([x_test_final[i] for i in range(len(y_test_final0)) if y_test_final0[i] == 0])
x_test_final2 = np.array([x_test_final[i] for i in range(len(y_test_final0)) if y_test_final0[i] == 1])


x_test_final1 = x_test_final1.reshape(-1, 28, 28, 3)

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


truelabel = np.load('true_label2.npy')
balanced_acc = balanced_accuracy_score(truelabel, y_test_final)
print("Number of differences: ", balanced_acc)

np.save('resultadotesk4.npy', y_test_final)
print(y_test_final.shape)