#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:33:39 2018

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


data = np.array(pd.read_csv('rollingsales_brooklyn-csv.csv', skiprows=1))
print(np.shape(data))

salesPrice = np.array(data[:,21], dtype=float)
grossSqFt = np.array(data[:,22], dtype=float)
landSqFt = np.array(data[:,23], dtype=float)
zipCode = np.array(data[:,10], dtype=float)
lot = np.array(data[:,5], dtype=float)
yearBuilt = np.array(data[:,16], dtype=float)

X = np.vstack((np.ones(np.size(salesPrice)), grossSqFt, landSqFt, zipCode, lot, yearBuilt)).T 
#print(X)
Y = salesPrice.reshape(np.size(salesPrice), 1)
#print(Y)
xtx = np.dot(X.T, X)
xty = np.dot(X.T, Y)
print(np.shape(xtx))
print(np.shape(xty))

b = np.linalg.solve(xtx, xty)
print("betas for Z: ", b)
xx = np.sort(grossSqFt)
xx2 = np.sort(landSqFt)
xx3 = np.sort(zipCode)
xx4 = np.sort(lot)
xx5 = np.sort(yearBuilt)
yy = np.array(b[0] + b[1]*xx + b[2]*xx2 + b[3]*xx3 + b[4]*xx4 + b[5]*xx5, dtype=float)
plt.title('predictors_year_built(x-axis) vs sales price predicted(y-axis)')
plt.plot(xx5, yy)
plt.scatter(yearBuilt,salesPrice,color='r')
plt.show()

plt.title('predictors_lot(x-axis) vs sales price predicted(y-axis)')
plt.plot(xx4, yy)
plt.scatter(lot,salesPrice,color='r')
plt.show()

plt.title('predictors_zipcode(x-axis) vs sales price predicted(y-axis)')
plt.plot(xx3, yy)
plt.scatter(zipCode,salesPrice,color='r')
plt.show()

plt.title('predictors_land_sqft(x-axis) vs sales price predicted(y-axis)')
plt.plot(xx2, yy)
plt.scatter(landSqFt,salesPrice,color='r')
plt.show()

plt.title('predictors_gross_sqft(x-axis) vs sales price predicted(y-axis)')
plt.plot(xx, yy)
plt.scatter(grossSqFt,salesPrice,color='r')
plt.show()

reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                         random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
temp = mean_squared_error(y_test, pred)
print("Mean square error for sales price model: ", temp)
#print("Zip code seems to be one of the best predictors")


################################
#KNN Neighbor
df = pd.read_csv('rollingsales_brooklyn.csv', skiprows=1)
estateData = np.array(np.array(df))
print(np.shape(estateData))
#print(estateData[10:13])
yString = np.array(estateData[:,1])#.reshape(np.size(estateData[:,1]),1) 
print(np.shape(yString))
yContainer = pd.factorize(yString)
#print(np.shape(yContainer))
#print(np.shape(yContainer[0]))
#print(yContainer[0])
y = np.array(yContainer[0]).reshape(np.size(yContainer[0]),1)
#print(y)
#print("&&&&&&&&&&")
#print('\n')
dataKNN = np.delete(estateData, [1,2,3,6,7,8,9,14,15,18,19,20], 1)

#print(np.shape(k))
#print(k[10:13])
dataMat = np.asmatrix(dataKNN)


X_train, X_test, y_train, y_test = train_test_split(
dataMat, y, test_size=0.33, random_state=42)


accuracy_array = []
k_array = []
for k in range(3,50,2):
    print(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    accuracy_array.append(accuracy.mean())
    k_array.append(k)
print (accuracy_array)
print(k_array)

class_error = 1.0 - np.array(accuracy_array)
plt.plot(k_array, class_error)
plt.xlabel('K')
plt.ylabel('Classification Error')
plt.show()

min_ind = np.argmin(class_error)
OptK = k_array[min_ind]
print ("Optimal value of K is %d " %  OptK)

knn = KNeighborsClassifier(n_neighbors=OptK)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print("Accuracy Score: ",accuracy_score(y_test, pred))










