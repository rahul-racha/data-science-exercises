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
import pandas as pd
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

#############################
#Linear Model Scikit
reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, 
                                                     random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('coefficients: ', reg.coef_)
print("Mean squared error: ",mean_squared_error(y_test, pred))



#################################
##build model with predictor Z and include x_1
meanSqErrorTraining = np.empty(50)
meanSqErrorTest = np.empty(50) 
reg = linear_model.LinearRegression()
for i in range(100, 5100, 100):
    N = i
    x_1 = np.random.normal(mean, stdDev, N)
    Z = np.square(x_1)
    #print(Z)
    trueError = np.random.normal(0, 2, N)
    y = trueBeta0 + trueBeta1 * x_1 + Z + trueError
    
    X = np.vstack((np.ones(N), x_1, Z)).T
    Y = y.T
#    xtx = np.dot(X.T, X)
#    xty = np.dot(X.T, Y)
#    
#    b = np.linalg.solve(xtx, xty)
#    print("betas for Z: ", b)
#    xmin = min(x_1)
#    xmax = max(x_1)
#    xx = np.linspace(xmin, xmax, 100)
#    xx2 = np.square(xx)
#    yy = np.array(b[0] + b[1]*xx + b[2]*xx2)
#    plt.plot(xx, yy)
#    plt.scatter(x_1,y,color='r')
#    plt.show()
    
    #print('\n')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, 
                                                         random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    #print('coefficients for TEST of sample size ',N," is ", reg.coef_)
    temp = mean_squared_error(y_test, pred)
    #print("Mean squared error for TEST of sample size:",N," is ",temp)
    np.append(meanSqErrorTest, temp)
    #if (temp < meanSqErrorTest):
    #   meanSqErrorTest = temp
    #print('\n')
    
    pred2 = reg.predict(X_train)
    #print('coefficients for TRAIN of sample size:',N," is ", reg.coef_)
    temp2 = mean_squared_error(y_train, pred2)
    #print("Mean squared error for TRAIN of sample size:",N," is ",temp2)
    np.append(meanSqErrorTraining, temp2)
    #if (temp2 < meanSqErrorTraining):
    #   meanSqErrorTraining = temp2  
    
    #print('\n')
    #print("*****************************")
fig, ax = plt.subplots()
lspace = np.array([x for x in range(100, 5100, 100)])
ax.plot(lspace, meanSqErrorTraining, 'go--', color='y', label='training - mean sq error')
ax.plot(lspace, meanSqErrorTest, 'k', markersize=12,color='b', label='test - mean sq error')
#plt.xlabel = "Sample size"
#plt.ylabel = "Mean square error"
plt.title('Model fitting predictor variables Z and x_1')
legend = ax.legend(loc='best', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()    
    
#################################
#build model that includes only x_1
meanSqErrorTraining = np.empty(50)
meanSqErrorTest = np.empty(50)
reg = linear_model.LinearRegression()
for i in range(100, 5100, 100):
    N = i
    x_1 = np.random.normal(mean, stdDev, N)
    trueError = np.random.normal(0, 2, N)
    y = trueBeta0 + trueBeta1 * x_1 + trueError
    X = np.vstack((np.ones(N), x_1)).T
    Y = y.T
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, 
                                                         random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    temp = mean_squared_error(y_test, pred)
    np.append(meanSqErrorTest, temp)
    
    pred2 = reg.predict(X_train)
    temp2 = mean_squared_error(y_train, pred2)
    np.append(meanSqErrorTraining, temp2)
    
fig, ax = plt.subplots()
lspace = np.array([x for x in range(100, 5100, 100)])
ax.plot(lspace, meanSqErrorTraining, 'go--', color='y', label='training - mean sq error')
ax.plot(lspace, meanSqErrorTest, 'k', markersize=12,color='b', label='test - mean sq error')
#plt.xlabel = "Sample size"
#plt.ylabel = "Mean square error"
plt.title('Model fitting x_1 only')
legend = ax.legend(loc='best', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()        















