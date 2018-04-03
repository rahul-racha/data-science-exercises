#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 03:23:56 2018

@author: rahulracha
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#histogram without error
#plt.hist(yNoError, density=True)
#xmin, xmax = plt.xlim()
#xVals = np.linspace(xmin, xmax, 100)
#yVals = mlab.normpdf(xVals, mean, stdDev)
#plt.plot(xVals, yVals, 'r', linewidth=2)
#plt.show()
mean, stdDev = 4, 7
N = 10
x_1 = np.random.normal(mean, stdDev, N)
#print(x_1)
trueError = np.random.normal(0, 2, N)
trueBeta0 = 1.1 
trueBeta1 = -8.2 
# generate data 
y = trueBeta0 + trueBeta1 * x_1 + trueError
yNoError = trueBeta0 + trueBeta1 * x_1
#print(y)

#draw histogram
print("HISTOGRAM")
plt.hist(y, density=True)
xmin, xmax = plt.xlim()
xVals = np.linspace(xmin, xmax, 100)
yVals = mlab.normpdf(xVals, mean, stdDev)
plt.plot(xVals, yVals, 'r', linewidth=2)
plt.show()

#scatterplot
plt.scatter(x_1, y)
plt.show()

#Recover Betas
X = np.vstack((np.ones(N) ,x_1)).T
Y = y.T
xtx = np.dot(X.T, X)
xty = np.dot(X.T, Y)

b = np.linalg.solve(xtx, xty)
print("betas: ", b)
xmin = min(x_1)
xmax = max(x_1)
ys = b[0] + b[1]*xmin
yl = b[0] + b[1]*xmax
print("REGRESSION MODEL")
plt.plot([xmin, xmax], [ys, yl])
plt.scatter(x_1, y)
plt.show()

#Linear Model Scikit
reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, 
                                                     random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('coefficients: ', reg.coef_)
print("Mean squared error: ",mean_squared_error(y_test, pred))

