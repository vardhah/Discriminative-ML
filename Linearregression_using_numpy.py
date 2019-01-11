# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:46:27 2018

@author: jvard
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import pandas as pd


def ComputeCost(X,Y, theta):
    cost = (X @ theta) - Y
    sqval = np.square(cost)
    sum = np.sum(sqval)
    return sum / len(Y)
    

def GradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros(num_iters)
    for itr in range(num_iters):
        theta = theta - (X.T @ (X @ theta - y))*(alpha/m)
        currentCost = ComputeCost(X, y, theta)
        j_history[itr] = currentCost
    #print(j_history)
    return theta


def NormalizeDataset(Z):
    maxval = Z.max()
    #mean = np.mean(Z)
    return (Z) / maxval
        

# Load CSV and columns
df = pd.read_csv("./DataSet/Housing.csv")
 
Y = df['price']
X = df['lotsize']  

ones = np.ones(len(X), int)
plt.scatter(X,Y)
alpha = .02
Y = Y.values
Y = Y.reshape(-1,1)
X = X.values
X = X.reshape(-1,1)

### Normalize the data set 
Xinput = NormalizeDataset(X)
Yinput = NormalizeDataset(Y)

input = np.column_stack((ones,Xinput))
theta = np.zeros((2,1))
## divide the dataset in training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    input, Yinput, test_size=0.33, random_state=42)

theta = GradientDescent(X_train,y_train,theta, alpha, 80000)
print(theta)
#testError = ComputeCost(X_test, y_test, theta)
## Plot data set and test model
plt.plot( X.max() * X_test[:,1], (X_test @ theta) * (Y.max()),color='black')





