import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

x = np.load('X.npy')
y = np.load('Y.npy')

sse = 0

#regg = 0 linear, regg = 1 Ridge, regg = 2 Lasso
regg = 1
alphaspace = np.logspace(-6, 3, 10000)
#alphaspace = [0.1,0.0871,1,10]

#Ridge
if regg == 1:
    reg = RidgeCV(alphas=alphaspace).fit(x, y)
    best_alpha = reg.alpha_
#Lasso
if regg == 2:
    reg = LassoCV(alphas = alphaspace, cv=15).fit(x, y)
    best_alpha = reg.alpha_

#Leave One Out
for i in range(len(x)):
    #Prepare Arrays
    xcopy = x.copy()
    testx = x[i:i+1]
    xcopy = np.delete(xcopy, i, axis=0)
    ycopy = y.copy()
    testy = y[i:i+1]
    ycopy = np.delete(ycopy, i, axis=0)
    #Linear Regression
    if regg == 0:
        clf = LinearRegression().fit(xcopy, ycopy)
    #Ridge
    if regg == 1:
        clf = Ridge(alpha=best_alpha).fit(xcopy, ycopy)
    #Lasso
    if regg == 2:
        clf = Lasso(alpha=best_alpha).fit(xcopy, ycopy)
    #Predict the output
    pred_y = clf.predict(testx)
    #print("pred" + str(pred_y))
    #print("test" + str(testy))
    sse += (testy - pred_y)**2 

print(best_alpha)
print(sse)
print(sse/len(x))


