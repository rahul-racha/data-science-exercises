#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:17:16 2018

@author: rahulracha
"""
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

start = datetime(2010, 4, 4)
end = datetime(2018, 4, 4)
aapl = web.DataReader('AAPL', 'yahoo', start, end)
aapl.index = pd.to_datetime(aapl.index)
aapl['Days'] = (aapl.index - aapl.index.min())/np.timedelta64(1,'D')
#aapl['Date'] = aapl['Date'].map(dt.datetime.toordinal)
print(aapl.head())
aapl['Log_Close'] = np.log(aapl['Close']/aapl['Close'].shift(1))
aapl['Log_Volume'] = np.log(aapl['Volume']/aapl['Volume'].shift(1))

aapl['Log_Close_Prev'] = aapl['Log_Close'].shift(1)
aapl['Log_Volume_Prev'] = aapl['Log_Volume'].shift(1)
aapl['Log_Close_Pred'] = aapl['Log_Close'].shift(-1)
aapl['Log_Volume_Pred'] = aapl['Log_Volume'].shift(-1)
#print(aapl.head())
#applCopy = aapl.copy()
days = np.array(aapl['Days'].tolist(), dtype=float)[3:-2]
#print(days)
logClose = np.array(aapl['Log_Close'].tolist())[3:-2]
logPrevClose = np.array(aapl['Log_Close_Prev'].tolist())[3:-2]
logPredClose = np.array(aapl['Log_Close_Pred'].tolist())[3:-2]

logVolume = np.array(aapl['Log_Volume'].tolist())[3:-2]
logPrevVolume = np.array(aapl['Log_Volume_Prev'].tolist())[3:-2]
logPredVolume = np.array(aapl['Log_Volume_Pred'].tolist())[3:-2]

#log Close returns
print("LOG PRICE RETURNS")
#print(np.shape(logClose))
#print(np.shape(logPrevClose))
#print(np.shape(logPredClose))
#print(np.shape(days))
X = np.vstack((np.ones(np.size(logClose)), days, logClose, logPrevClose)).T
Y = logPredClose.T
xtx = np.dot(X.T, X)
xty = np.dot(X.T, Y)

b = np.linalg.solve(xtx, xty)
print("betas for Z: ", b)
xmin = min(days)
xmax = max(days)
xx = days#np.linspace(xmin, xmax, 100)
xx2 = logClose
xx3 = logPrevClose
yy = np.array(b[0] + b[1]*xx + b[2]*xx2 + b[3]*xx3, dtype=float)
plt.title('days(x-axis) vs log returns predicted(y-axis)')
plt.plot(xx, yy)
plt.scatter(days,logPredClose,color='r')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                         random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
temp = mean_squared_error(y_test, pred)
print("Mean square error for log returns model: ", temp)
#print(np.shape(X_test))
#print(X_test[-1])
t = datetime.strptime('2018-04-05', '%Y-%m-%d')
d = np.array([1, (t - aapl.index.min())/np.timedelta64(1,'D'), logClose[-1], 
              logPrevClose[-1]]).reshape(4,1).T
print("*********************************")
print("Log return on April 5th 2018 is ",reg.predict(d))


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#log Close returns
print("LOG VOLUME RETURNS")
X = np.vstack((np.ones(np.size(logVolume)), days, logVolume, logPrevVolume)).T
Y = logPredVolume.T
xtx = np.dot(X.T, X)
xty = np.dot(X.T, Y)

b = np.linalg.solve(xtx, xty)
print("betas for Z: ", b)
xmin = min(days)
xmax = max(days)
xx = days#np.linspace(xmin, xmax, 100)
xx2 = logVolume
xx3 = logPrevVolume
yy = np.array(b[0] + b[1]*xx + b[2]*xx2 + b[3]*xx3, dtype=float)
plt.title('days(x-axis) vs log volume returns predicted(y-axis)')
plt.plot(xx, yy)
plt.scatter(days,logPredVolume,color='r')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                         random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
temp = mean_squared_error(y_test, pred)
print("Mean square error for log returns model: ", temp)

t = datetime.strptime('2018-04-05', '%Y-%m-%d')
d = np.array([1, (t - aapl.index.min())/np.timedelta64(1,'D'), logVolume[-1], 
              logPrevVolume[-1]]).reshape(4,1).T
print("*********************************")
print("Log return on April 5th 2018 is ",reg.predict(d))


