# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 08:53:11 2019

@author: jvard
"""

# classification algorithm 

import math
import numpy as np

def sigmoid(x):
    result = np.zeros(len(x))
    result = result.reshape(-1,1)
    for i in range(len(x)):
        val = 1 / (1 + math.exp(-x[i]))
        result[i] = val # 1 if val >= 0.5 else 0
    return result

def costfunction(X, y, theta):
    datasetCount = len(y)
    return ((y.T @ np.log(sigmoid(X @ theta))) + ((1-y).T @ np.log(1 - sigmoid(X @ theta)))) * (1/datasetCount)
    

def GradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    for itr in range(num_iters):
        theta = theta - (X.T @ (sigmoid(X @ theta) - y))*(alpha/m)
       # cost = costfunction(X,y, theta)
        #print(cost)
    #print(j_history)
    return theta

def predictLabel(X, theta):
    return 1 if sigmoid(X @theta) > 0.5 else 0

X = np.array([[1,4.85,9.63],[1,8.62,3.23],[1,5.43,8.23],[1,9.21,6.34]])
y = np.array([[1],[0],[1],[0]])

theta = np.array([[0],[0],[0]])

theta = GradientDescent(X,y,theta, 0.5,10)

print(theta)

testdata = np.array([[1,8.62,3.23]])
predictedval = predictLabel(testdata, theta)
print("The predicted class is: ", predictedval)