# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 08:53:11 2019

@author: jvard
"""

# classification algorithm 

import math
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

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
    
def NormalizeDataset(Z):
    maxval = Z.max()
    #mean = np.mean(Z)
    return (Z) / maxval


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

def performLgRegression(path,targetattribute, featureattributes):
    # read the data set 
    df = pd.read_csv(path)
    
    # perform this task only ur dataset contains string value (yes / no)
    Y  = df[targetattribute].map({'yes': 1, 'no': 0})

    featureattributes = featureattributes.split(',')
    X = df[featureattributes]

    ### Normalize the data set 
    X = NormalizeDataset(X)
    
    ones = np.ones(X.shape[0], int).reshape(-1,1)
    
    alpha = .02
    Y = Y.values
    Y = Y.reshape(-1,1)

    input = np.column_stack((ones,X))

    theta = np.zeros((X.shape[1] + 1,1))
    
     ## divide the dataset in training / testing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        input, Y, test_size=0.33, random_state=42)

    theta = GradientDescent(X_train,y_train,theta, alpha,10)
    print(theta)

if __name__ == "__main__":
    path = sys.argv[1]
    targetattribute = sys.argv[2]
    featureattributes = sys.argv[3]
    print("path to dataSet: ", path)
    print("Feature attribute: ", featureattributes)
    print("Target attribute: ", targetattribute)
    logisticcoefficient = performLgRegression(path, targetattribute, featureattributes)
