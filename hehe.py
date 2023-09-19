import numpy as np
from sklearn.linear_model import LinearRegression

x = np.load('X.npy')
y = np.load('Y.npy')

sse = 0
score = 0
for i in range(len(x)):
    xcopy = x
    testx = x[i]
    xcopy = np.delete(x, i)
    ycopy = y
    testy = y[i]
    ycopy = np.delete(y, i)
    #Linear Regression
    reg = LinearRegression().fit(xcopy, ycopy)
    predy = reg.predict(testx)
    beta = reg.coef_
    sse += (testy - predy*beta)**2 
    


#
#reg = LinearRegression().fit(x, y)
#
#score = reg.score(x, y)
#print(score)
#beta = reg.coef_
#
#pred = reg.predict(np.load('X_test_regression1.npy'))
#
#print(pred)

