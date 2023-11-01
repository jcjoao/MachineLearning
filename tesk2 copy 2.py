import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

def model(x,y,type):
    if type == 0:
        return LinearRegression().fit(x, y)
    if type == 1:
        alphaspace = np.logspace(-14, 1, 10000)
        reg = RidgeCV(alphas=alphaspace).fit(x, y)
        print("Best alpha:  " + str(reg.alpha_))
        return reg
    if type == 2:
        alphaspace = np.logspace(-4, 3, 1000)
        return LassoCV(alphas = alphaspace, cv=len(x)).fit(x, y.ravel())

def leave_one_out(best_alpha, regg,x,y):
    sse = 0
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
        #Elastic Net
        if regg == 3:
            clf = ElasticNet(alpha=best_alpha, l1_ratio=1).fit(xcopy, ycopy)
        #Predict the output
        pred_y = clf.predict(testx)
        sse += (testy - pred_y)**2

    return sse

#Load data
x = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# 0 is Linear, 1 is Ridge, 2 is Lasso
model_type = 0

#Initial Regression With Everything
reg = LinearRegression().fit(x, y)
pred_y = reg.predict(x)
sse = ((y - pred_y)**2)

sses = [0.2, 0.05, 0.01]
color1 = ['darkblue', 'blue', 'purple']
color2 = ['firebrick', 'red', 'orange']
color3= ['darkgreen', 'green', 'lightgreen']
for j in range(3):
    #Remove outliers
    outliers = np.where(sse > sses[j])[0]
    x1 = np.delete(x.copy(), outliers, axis=0)
    y1 = np.delete(y.copy(), outliers, axis=0)
    x2 = x[outliers]
    y2 = y[outliers]
    reg1 = model(x1,y1,model_type)
    reg2 = model(x2,y2,model_type)

    size1 = [len(x1)]
    size2 = [len(x2)]
    trades_plot = []
    trades_aux = np.array([])

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
        #Elements that might need to be traded
        trades = np.where(sse1 > sse2)[0]



        #Trade
        nx1 = len(x1)
        x1 = np.delete(x.copy(), trades, axis=0)
        y1 = np.delete(y.copy(), trades, axis=0)
        x2 = x[trades]
        y2 = y[trades]

        # print(trades)
        size1.append(len(x1))
        size2.append(len(x2)) 
        equals = len([i for i in trades if i in trades_aux])
        trades_plot.append(len(trades) + len(trades_aux) - 2*equals)
        trades_aux = trades.copy()


        reg1 = model(x1,y1,model_type)
        reg2 = model(x2,y2,model_type)



    ####################################################


    
    #plot size of x1 and x2
    size_axis = [i for i in range(len(size1))]
    # plt.plot(size_axis,size1, color=color1[j])
    # plt.plot(size_axis,size2, color=color2[j])
    # plt.title("Progression of the size of the sets and number of trades between them")
    # plt.show()

    #shift x axis to the right
    trades_axis = [i+0.325+j*0.15 for i in range(len(trades_plot))]
    #plot number of trades
    plt.bar(trades_axis,trades_plot, color=color3[j], width=0.15)
    
    # plt.xticks(x_ticks_positions, x_ticks_labels)


num_ticks = 11
x_ticks_positions = np.linspace(0, 10, num_ticks)
x_ticks_labels = [str(int(x)) for x in x_ticks_positions]
plt.xticks(x_ticks_positions, x_ticks_labels)
plt.xlabel('Iteration')
plt.ylabel('Number of trades')
# plt.xlabel('Iteration')
# plt.ylabel('Size of the sets')

plt.show()


###########################



pred_y1 = reg1.predict(x1)
sse1 = ((y1 - pred_y1)**2)
pred_y2 = reg2.predict(x2)
sse2 = ((y2 - pred_y2)**2)
print("-----------------------------")
print("Final Division")
print("SIZE OF X1: " + str(x1.shape) + "SSE1" + str(np.sum(sse1)))
print("SIZE OF X2: " + str(x2.shape) + "SSE2" + str(np.sum(sse2)))
print("-----------------------------")

#-------------------------------------------------------------------------------------------------------------------
