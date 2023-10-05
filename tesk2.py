import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

def model(x,y,type):
    if type == 0:
        return LinearRegression().fit(x, y)
    if type == 1:
        alphaspace = np.logspace(-4, 3, 1000)
        return RidgeCV(alphas=alphaspace).fit(x, y)
    if type == 2:
        alphaspace = np.logspace(-4, 3, 1000)
        #print("x is " + str(x))
        #print("y is " + str(y))
        #print(len(x))
        return LassoCV(alphas = alphaspace, cv=len(x)).fit(x, y.ravel())

#Load data
x = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# 0 is Linear, 1 is Ridge, 2 is Lasso
model_type = 1

cluster = 0

#Initial Regression With Everything
reg = LinearRegression().fit(x, y)
pred_y = reg.predict(x)
sse = ((y - pred_y)**2)

#Remove outliers
outliers = np.where(sse > 0.2)[0]
x1 = np.delete(x.copy(), outliers, axis=0)
y1 = np.delete(y.copy(), outliers, axis=0)
x2 = x[outliers]
y2 = y[outliers]
print("SIZE OF X1: " + str(x1.shape))
print("SIZE OF X2: " + str(x2.shape))

#reg1 = LinearRegression().fit(x1, y1)
#reg2 = LinearRegression().fit(x2, y2)
reg1 = model(x1,y1,model_type)
reg2 = model(x2,y2,model_type)

for i in range(10):
    #Prediction of y1
    if model_type == 2:
        pred_ = reg1.predict(x)
        pred_y1 = pred_.reshape((len(x),1))
    else:
        pred_y1 = reg1.predict(x)
    sse1 = ((y - pred_y1)**2)
    #Prediction of y2
    if model_type == 2:
        pred_ = reg2.predict(x)
        pred_y2 = pred_.reshape((len(x),1))
    else:
        pred_y2 = reg2.predict(x)
    sse2 = ((y - pred_y2)**2)
    print("SSE1 " + str(np.sum(((y1 - reg1.predict(x1))**2))))
    print("SSE2 " + str(np.sum(((y2 - reg2.predict(x2))**2))))
    #Elements that might need to be traded
    trades = np.where(sse1 > sse2)[0]
    #print("this is x2" + str(trades))
    #Trade
    nx1 = len(x1)
    x1 = np.delete(x.copy(), trades, axis=0)
    y1 = np.delete(y.copy(), trades, axis=0)
    x2 = x[trades]
    y2 = y[trades]
    #print("SIZE OF X1: " + str(x1.shape))
    #print("SIZE OF X2: " + str(x2.shape))
    #print("number of traded elements =" + str(nx1 - len(x1)))
    reg1 = model(x1,y1,model_type)
    reg2 = model(x2,y2,model_type)

pred_y1 = reg1.predict(x1)
sse1 = ((y1 - pred_y1)**2)
pred_y2 = reg2.predict(x2)
sse2 = ((y2 - pred_y2)**2)
print("-----------------------------")
print("SIZE OF X1: " + str(x1.shape))
print("SSE1 HEHE " + str(np.sum(sse1)))
print("SIZE OF X2: " + str(x2.shape))
print("SSE2 HEHE " + str(np.sum(sse2)))
print(trades)

if cluster:    
    # Assuming you have 'x' and 'y' as your data
    # Stack 'x' and 'y' horizontally to create a single feature matrix
    newx = np.column_stack((x, y))
    # Choose the number of clusters (in this case, 2)
    n_clusters = 2
    gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(newx)
    cluster_assignments = gmm.predict(newx)
    # Split your data into two sets based on the cluster assignments
    x_1 = newx[cluster_assignments == 0]
    x_2 = newx[cluster_assignments == 1]

    tradesv2 = np.where(cluster_assignments == 1)[0]
    print(tradesv2)

    difference = np.setxor1d(trades, tradesv2)
    print(difference)