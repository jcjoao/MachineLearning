import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet


def leave_one_out(best_alpha, regg,x):
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
x = np.load('X_train_regression1.npy')
y = np.load('y_train_regression1.npy')

# strat = 0 Manual calculation of the best alpha
# strat = 1 Using CV functions from sklearn, tries every method and tells the best one for our data
strat = 1
#Space of possible values of alpha
alphaspace = np.logspace(-2, 2, 1000)

if strat == 0:
    alpha_scores1 = np.zeros(1000)
    alpha_scores2 = np.zeros(1000)
    j=0
    for best_alpha in alphaspace:
        sse1 = leave_one_out(best_alpha,regg=1,x=x)
        sse2 = leave_one_out(best_alpha,regg=2,x=x)
        alpha_scores1[j] = sse1/len(x)
        alpha_scores2[j] = sse2/len(x)
        j+=1

    print("------------------------------")
    print("Best alpha for Ridge:  " + str(alphaspace[np.where(alpha_scores1 == min(alpha_scores1))[0][0]]))
    print("MSE value:  " + str(min(alpha_scores1)))
    print("SSE value:  " + str(min(alpha_scores1)*len(x)))

    print("------------------------------")
    print("Best alpha for Lasso:  " + str(alphaspace[np.where(alpha_scores2 == min(alpha_scores2))[0][0]]))
    print("MSE value:  " + str(min(alpha_scores2)))
    print("SSE value:  " + str(min(alpha_scores2)*len(x)))

    plt.semilogx(alphaspace, alpha_scores1)
    plt.xlabel('Alpha Values')
    plt.ylabel('Alpha Scores')
    plt.title('Alpha Evolution')
    plt.show()

if strat == 1:
    #Ordinary Least Squares
    sse = leave_one_out(0, regg=0,x=x)
    print("------------------------------")
    print("Using Ordinary Least Squares")
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(x)))
    ssearray = np.array([sse/len(x)])

    #Ridge
    reg1 = RidgeCV(alphas=alphaspace).fit(x, y)
    best_alpha = reg1.alpha_

    sse = leave_one_out(best_alpha, regg=1,x=x)
    print("------------------------------")
    print("Using Ridge")
    print("Best alpha:  " + str(best_alpha))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(x)))
    ssearray = np.append(ssearray,sse/len(x))

    #Lasso
    reg2 = LassoCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
    best_alpha = reg2.alpha_

    sse = leave_one_out(best_alpha, regg=2,x=x)
    print("------------------------------")
    print("Using Lasso")
    print("Best alpha:  " + str(best_alpha))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(x)))
    ssearray = np.append(ssearray,sse/len(x))

    #Elastic Net
    reg3 = ElasticNetCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
    best_alpha = reg3.alpha_
    best_ratio = reg3.l1_ratio_

    sse = leave_one_out(best_alpha, regg=3,x=x)
    print("------------------------------")
    print("Using ElasticNet")
    print("Best alpha:  " + str(best_alpha))
    print("Best ratio:  " + str(best_ratio))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(x)))
    ssearray = np.append(ssearray,sse/len(x))

    #Feature Selection
    reg = LassoCV(alphas = alphaspace, cv=15).fit(x, y.ravel())
    #print(reg.coef_)
    useless = np.where(reg.coef_ == 0.0)
    #print(useless)
    xgood = x.copy()
    xgood = np.delete(xgood, useless, axis=1)

    #Feature Selection with Linear
    sse = leave_one_out(0,regg=0,x=xgood)
    print("------------------------------")
    print("Using Feature Selection and Ordinary Least Squares")
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(xgood)))
    ssearray = np.append(ssearray,sse/len(xgood))

    #Feature Selection with Ridge
    reg = RidgeCV(alphas=alphaspace).fit(xgood, y)
    best_alpha_fs_ridge = reg.alpha_
    sse = leave_one_out(best_alpha_fs_ridge,regg=1,x=xgood)
    print("------------------------------")
    print("Using Feature Selection and Ridge")
    print("Best alpha:  " + str(best_alpha_fs_ridge))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(xgood)))
    ssearray = np.append(ssearray,sse/len(xgood))
    
    #Feature Selection with Lasso
    reg = LassoCV(alphas=alphaspace, cv=15).fit(xgood, y.ravel())
    best_alpha_fs_lasso = reg.alpha_
    sse = leave_one_out(best_alpha_fs_lasso,regg=2,x=xgood)
    print("------------------------------")
    print("Using Feature Selection and Lasso")
    print("Best alpha:  " + str(best_alpha_fs_lasso))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(xgood)))
    ssearray = np.append(ssearray,sse/len(xgood))

    #Feature Selection with Lasso
    reg = ElasticNetCV(alphas = alphaspace, cv=15).fit(xgood, y.ravel())
    best_alpha_fs_elastic = reg.alpha_
    best_ratio = reg.l1_ratio_
    sse = leave_one_out(best_alpha_fs_elastic,regg=3,x=xgood)
    print("------------------------------")
    print("Using Feature Selection and Elastic Net")
    print("Best alpha:  " + str(best_alpha_fs_elastic))
    print("SSE value:  " + str(sse))
    print("MSE value:  " + str(sse/len(xgood)))
    ssearray = np.append(ssearray,sse/len(xgood))

    #Tell the best one
    print("------------------------------")
    my_list = ["Linear", "Ridge", "Lasso", "Elastic Net", "Feature Selection with linear", "Feature Selection with Ridge"]
    my_list += ["Feature Selection with Lasso","Feature Selection with ElasticNet"]
    print("The Best Method is: " + my_list[np.argmin(ssearray)])
    print("With an MSE of: " + str(min(ssearray)))
    print("And a SSE of: " + str(min(ssearray)*len(xgood)))
    print()

    #FINAL RESULT knowing Feature Selection with Ridge is the best method
    xtest = np.load("X_test_regression1.npy")
    xtest = np.delete(xtest, useless, axis=1)
    reg = Ridge(alpha=best_alpha_fs_ridge).fit(xgood, y)
    ytest = reg.predict(xtest)
    #print(ytest)
    np.save('result',ytest)
