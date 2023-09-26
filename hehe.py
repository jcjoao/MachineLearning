import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet


def leave_one_out(regg):
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
            clf = ElasticNet(alpha=best_alpha, l1_ratio=best_ratio).fit(xcopy, ycopy)
        #Predict the output
        pred_y = clf.predict(testx)
        sse += (testy - pred_y)**2

    return sse

#Load data
x = np.load('X.npy')
y = np.load('Y.npy')

# strat = 0 Manual calculation of the best alpha
# strat = 1 Using CV functions from sklearn
# strat = 2 Using Lasso for Feature Selection, and then Ridge
strat = 2
#regg = 0 linear, regg = 1 Ridge, regg = 2 Lasso, regg = 3 Elastic Net
regg = 1
#Space of possible values of alpha
alphaspace = np.logspace(-2, 2, 1000)

if strat == 0:
    if regg == 3: exit()
    alpha_scores = np.zeros(1000)
    j=0
    for best_alpha in alphaspace:
        sse = leave_one_out(regg)
        alpha_scores[j] = sse/len(x)
        j+=1
        
    print("Best alpha:  " + str(alphaspace[np.where(alpha_scores == min(alpha_scores))[0][0]]))
    print("SSE value:  " + str(min(alpha_scores)))

    plt.semilogx(alphaspace, alpha_scores)
    plt.xlabel('Alpha Values')
    plt.ylabel('Alpha Scores')
    plt.title('Alpha Evolution')
    plt.show()

if strat == 1:
    #Ridge
    if regg == 1:
        reg = RidgeCV(alphas=alphaspace).fit(x, y)
        best_alpha = reg.alpha_
    #Lasso
    if regg == 2:
        reg = LassoCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
        best_alpha = reg.alpha_
    #Elastic Net
    if regg == 3:
        reg = ElasticNetCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
        best_alpha = reg.alpha_
        best_ratio = reg.l1_ratio_

    sse = leave_one_out(regg)

    print("Best alpha:  " + str(best_alpha))
    print("SSE value:  " + str(sse/len(x)))


if strat == 2:
    reg = LassoCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
    print(reg.coef_)
    useless = np.where(reg.coef_ == 0.0)
    print(useless)
    xcopy = x.copy()
    xcopy = np.delete(xcopy, useless, axis=1)

    reg = RidgeCV(alphas=alphaspace).fit(xcopy, y)
    best_alpha = reg.alpha_
    #Leave one out with Ridge
    sse = leave_one_out(1)
    print("Best alpha:  " + str(best_alpha))
    print("SSE value:  " + str(sse/len(x)))
