import numpy as np
import copy

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import tree


def bandit_non_stationarity(data):
    # perform linear regression
    regr = linear_model.LinearRegression()

    X = []
    y = []
    for ind in range(len(data)):
        # adding 1 for bias column
        X.append([1,data[ind][0]])
        y.append(data[ind][1])
    X, y = np.array(X), np.array(y)
    regr.fit(X, y)
    residuals = np.absolute(y - regr.predict(X))

    return residuals[-1]


def factored_bandit_distributional(data):
    # if data is flagged, simply return 0 as the score doesn't matter
    if data == 'flag':
        return 0
    X = []
    y = []

    regr = linear_model.LinearRegression()
    for ind in range(len(data)):
        # adding 1 for bias column
        X.append([1, 1 if data[ind][0][0]==1 and data[ind][0][1]==1 else 0, 1 if data[ind][0][1] == 0 else 0])
        y.append(data[ind][1])
    X, y = np.array(X), np.array(y)
    
    if np.linalg.matrix_rank(X) < 3:
        if not (np.all(X[:,1] == np.ones(len(y))) or np.all(X[:,1] == np.zeros(len(y)))):
            X = copy.deepcopy(X[:,:2])
        else:
            return 0.
    regr.fit(X, y)
    y_pred = regr.predict(X)
    denom1 = np.sqrt(mean_squared_error(y, y_pred)*len(y)/(len(y)-3))
    denom2 = np.sqrt(np.linalg.inv(X.T@X)[1][1])
    numerator = regr.coef_[1]
    return np.absolute(numerator/(denom1*denom2))


def contextual_bandit_conditional_independence(data):
    '''This function is used both for hypothesis testing 
    and confidence interval construction.
    The b in the input corresponds to confidence interval. As default, b is set
    to 0, in which case it is just regular testing'''
    X = []
    y = []
    regr = linear_model.LinearRegression()
    for ind in range(len(data)):
        # adding 1 for bias column
        X.append([1, data[ind][0][0], data[ind][0][1][0], data[ind][0][1][1]])
        y.append(data[ind][1])
    X, y = np.array(X), np.array(y)
    
    if np.linalg.matrix_rank(X) < 4:
        return 0.

    regr.fit(X, y)
    y_pred = regr.predict(X)
    denom1 = np.sqrt(mean_squared_error(y, y_pred)*len(y)/(len(y)-4))
    denom2 = np.sqrt(np.linalg.inv(X.T@X)[1][1])
    numerator = regr.coef_[1]
    return np.absolute(numerator/(denom1*denom2))


def mdp_non_stationarity(data):
    # if data is flagged, simply return 0 as the score doesn't matter
    if data == 'flag':
        return 0

    X = []
    y = []
    # exclude the last action timestep
    for ind in range(len(data)-1):
        # adding 1 for bias column
        X.append([data[ind][0][0],data[ind][0][1]])
        y.append(data[ind][1])

    X, y = np.array(X), np.array(y)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    # calculating log likelihood
    [p] = clf.predict_proba([X[-1]])

    # return log likelihood using regr.classes_
    return -np.log(p[list(clf.classes_).index(y[-1])])


def contextual_bandit_non_stationarity(data):
    # perform linear regression
    #regr = linear_model.LinearRegression()
    #regr = linear_model.Lasso()
    regr = linear_model.LassoCV(cv=5,max_iter=10000)

    X = []
    y = []
    for ind in range(len(data)):
        # adding 1 for bias column
        X.append(np.append([1, data[ind][0][0]], data[ind][0][1]))
        y.append(data[ind][1])
    X = np.vstack(X)
    y = np.array(y)
    
    regr.fit(X, y)
    residuals = np.absolute(y - regr.predict(X))

    return residuals[-1]