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

### sigmoidal function f(z) = 1/ 1 + e^z 
def sigmoid(x):
    result = np.zeros(len(x))
    result = result.reshape(-1,1)
    for i in range(len(x)):
        val = 1 / (1 + math.exp(-x[i]))
        result[i] = val # 1 if val >= 0.5 else 0
    return result

### cost for for logistic regression follow below link for more detail 
### https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
def costfunction(X, y, theta):
    datasetCount = len(y)
    return ((y.T @ np.log(sigmoid(X @ theta))) + ((1-y).T @ np.log(1 - sigmoid(X @ theta)))) * (1/datasetCount)

## when features are of different range, we use normalized function to bring data set in normalized form 
def NormalizeDataset(Z):
    maxval = Z.max()
    #mean = np.mean(Z)
    return (Z) / maxval

### this function calculate the rate at which paramter descent
def GradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros(num_iters)
    for itr in range(num_iters):
        theta = theta - (X.T @ (sigmoid(X @ theta) - y))*(alpha/m)
        cost = costfunction(X,y, theta)
        j_history[itr] = cost
    #print(j_history)
    return theta

## predictive model which use trained theta and use that to predict new label
def predictLabel(X, theta):
    predictedlabel = sigmoid(X @ theta)
    new_predictedlabel = [1 if el> .5 else 0 for el in predictedlabel]
    predictedlabel = np.array(new_predictedlabel)
    return predictedlabel.reshape(-1,1)
    
## This function predict the model accuracy with trained theta
def findModelAccuracy(theta, X , Y):
    predictedLabel = predictLabel(X, theta)
    resultantVector = (predictedLabel - Y)
    totalerror = np.sum(np.absolute(resultantVector))
    return ((Y.shape[0] - totalerror) / Y.shape[0]) * 100
    
### Main method to generate theta and check accuracy of model on test dataset   
def performLgRegression(X,Y,learningrate,no_iteration):
    theta = np.zeros((X.shape[1],1))
    theta = GradientDescent(X,Y,theta, learningrate,no_iteration)
    return theta

## module responsible for training of model and tweeking the model to improve accuracy
def TrainModel(path,targetattribute, featureattributes):
    # read the data set 
    df = pd.read_csv(path)
    
    # perform this task only ur dataset contains string value (yes / no)
    Y  = df[targetattribute].map({'yes': 1, 'no': 0})

    featureattributes = featureattributes.split(',')
    X = df[featureattributes]

    ### Normalize the data set 
    X = NormalizeDataset(X)
    
    ones = np.ones(X.shape[0], int).reshape(-1,1)
    
    alpha = 1
    Y = Y.values
    Y = Y.reshape(-1,1)

    input = np.column_stack((ones,X))
    
     ## divide the dataset in training / testing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        input, Y, test_size=0.33, random_state=42)
    
    trainedTheta = performLgRegression(X_train,y_train,alpha,500)
    modelAccuracy = findModelAccuracy(trainedTheta,X_test,y_test)
    print('Model Accuracy: ',modelAccuracy)
    
    
if __name__ == "__main__":
    path = sys.argv[1]
    targetattribute = sys.argv[2]
    featureattributes = sys.argv[3]
    print("path to dataSet: ", path)
    print("Feature attribute: ", featureattributes)
    print("Target attribute: ", targetattribute)
    ## call the method to train the model
    TrainModel(path, targetattribute, featureattributes)
