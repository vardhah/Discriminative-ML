# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:05:59 2019

@author: jvard
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

def ComputeCost(X,Y, theta):
    cost = (X @ theta) - Y
    sqval = np.square(cost)
    sum = np.sum(sqval) + np.sum(np.square(theta))
    return sum / len(Y)
    

def GradientDescent(X, y, theta, alpha, num_iters, regfactor):
    m = len(y)
    j_history = np.zeros(num_iters, 'int')
    j_history = j_history.reshape(-1,1)
    #print(j_history)
    for itr in range(num_iters):
        theta = theta - (alpha/m)*((X.transpose() @ (X @ theta - y)) + regfactor*theta)
        j_history[itr] = ComputeCost(X, y, theta);
    return theta
        

x = [1,2,3,4,5,6,7,8,9,10]   #### this is single variable feature value
y = [2.5,3,4,6,5,9,6,8.5,9,10] ### this is the level of the variable.

ones = np.ones(len(x), int)
X = np.asarray(x)
Y = np.asarray(y)
Y= Y.reshape(-1,1)
learningRate = .01
regfactor = 1
input = np.column_stack((ones,X))
theta = np.zeros((2,1),  dtype=int)

## divide the dataset in training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    input, Y, test_size=0.33, random_state=42)

theta = GradientDescent(X_train,y_train,theta,learningRate, 50,regfactor)
#plt.scatter(X_train[:,1],y_train)
print(theta)
trainingError = ComputeCost(X_train, y_train, theta)
testError = ComputeCost(X_test, y_test, theta)
print("TrainingError", trainingError)
print("TestError", testError)
print('---- Deviation from predicted value to Test label-----')
plt.scatter(X_test[:,1],(X_test @ theta) - y_test)









