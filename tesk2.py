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

#Remove outliers
outliers = np.where(sse > 0.2)[0]
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

for i in range(5):
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


size_axis = [i for i in range(len(size1))]

num_ticks = 6
x_ticks_positions = np.linspace(0, 5, num_ticks)
x_ticks_labels = [str(int(x)) for x in x_ticks_positions]

#plot size of x1 and x2
plt.plot(size_axis,size1, color='blue')
plt.plot(size_axis,size2, color='red')
plt.xticks(x_ticks_positions, x_ticks_labels)
plt.xlabel('Iteration')
plt.ylabel('Size of the sets')
# plt.title("Progression of the size of the sets and number of trades between them")
plt.show()

#shift x axis to the right
trades_axis = [i+0.5 for i in range(len(trades_plot))]
#plot number of trades
plt.bar(trades_axis,trades_plot, color='green', width=0.35)
plt.xlabel('Iteration')
plt.ylabel('Number of trades')
plt.xticks(x_ticks_positions, x_ticks_labels)
plt.show()



x1index = [i for i in range(len(x)) if x[i] in x1]
x2index = [i for i in range(len(x)) if x[i] in x2]

#plot wich points are in which division
fp1 = [1 if x in x1index else 0 for x in range(len(x))]
fp2 = [1 if x in x2index else 0 for x in range(len(x))]
xaxis = [i for i in range(len(x))]
#bar    plot
subplot = plt.subplot(2,1,1)
# subplot.title("Division of points")
subplot.bar(xaxis,fp1, color='blue')
subplot.bar(xaxis,fp2, color='red')
ax = plt.gca()
#hide y-axis 
ax.get_yaxis().set_visible(False)
# plt.title("Division of points between the sets/models")
num_ticks = 21
x_ticks_positions = np.linspace(0, 100, num_ticks)
x_ticks_labels = [str(int(x)) for x in x_ticks_positions]
plt.xticks(x_ticks_positions, x_ticks_labels)
####################################################



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
#GMM  
# Stack x and y together
newx = np.column_stack((x, y))
n_clusters = 2
gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(newx)
cluster_assignments = gmm.predict(newx)
# Split your data into two sets based on the cluster assignments
x_1 = x[cluster_assignments == 0]
x_2 = x[cluster_assignments == 1]
y_1 = y[cluster_assignments == 0]
y_2 = y[cluster_assignments == 1]
x_1index = np.where(cluster_assignments == 0)[0]
x_2index = np.where(cluster_assignments == 1)[0]
#print(x_2index)
difference = np.setxor1d(x2index, x_2index)
print("Diferent Points between Manual and GMM:" + str(difference))
print("-----------------------------")



########################################################
#plot wich points are in which division
fs1 = [1 if x in x_1index else 0 for x in range(len(x))]
fs2 = [1 if x in x_2index else 0 for x in range(len(x))]
xaxis = [i for i in range(len(x))]
#bar plot
subplot = plt.subplot(2,1,2)
subplot.bar(xaxis,fs1, color='blue')
subplot.bar(xaxis,fs2, color='red')

ax = plt.gca()

#hide y-axis 
ax.get_yaxis().set_visible(False)

########################################################



plt.xticks(x_ticks_positions, x_ticks_labels)
plt.xlabel('Sample index in input data') 

plt.show()


#-----------------------------
#Remove different elements from x1 and x2
x2index_final = [x for x in x2index if x not in difference]
x2_final = x[x2index_final]
y2_final = y[x2index_final]
print("SIZE OF X2: " + str(x2_final.shape))

x1index_final = [x for x in x1index if x not in difference]
x1_final = x[x1index_final]
print("SIZE OF X1: " + str(x1_final.shape))
y1_final = y[x1index_final]


#-------------------------------------------------------------------------------------------------------------------
#FINAL RESULT knowing Ridge is the best
xtest = np.load("X_test_regression2.npy")

reg1 = RidgeCV(alphas=np.logspace(-1,0,1000)).fit(x1_final, y1_final)
reg1 = Ridge(alpha=reg1.alpha_).fit(x1_final, y1_final)
ytest1 = reg1.predict(xtest)

reg2 = LinearRegression().fit(x2_final, y2_final)
ytest2 = reg2.predict(xtest)

y_out = np.column_stack((ytest1,ytest2))

print(y_out.shape)
print(y_out)

np.save('result2',y_out)
#-----------------------------
print(trades_plot)