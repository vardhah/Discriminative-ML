# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:16:46 2019

@author: vardhah
"""

import os
import numpy as np
import struct
from matplotlib import pyplot


path = 'C:/Users/HPP/Desktop/Python/DL clas/MNIST'   # the training set is stored in this directory

# Train data
fname_train_images = os.path.join(path, 'train-images.idx3-ubyte')  # the training set image file path
fname_train_labels = os.path.join(path, 'train-labels.idx1-ubyte')  # the training set label file path

#Test data
fname_test_images = os.path.join(path, 't10k-images.idx3-ubyte')  # the training set image file path
fname_test_labels = os.path.join(path, 't10k-labels.idx1-ubyte')  # the training set label file path
    
# open the label file and load it to the "train_labels"
with open(fname_train_labels, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    train_labels = np.fromfile(flbl, dtype=np.uint8)

# open the image file and load it to the "train_images"
with open(fname_train_images, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    train_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_labels), rows, cols)
    
    
with open(fname_test_labels, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    test_labels = np.fromfile(flbl, dtype=np.uint8)

# open the image file and load it to the "train_images"
with open(fname_test_images, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    test_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_labels), rows, cols)
    
#print('The training set contains', len(train_images), 'images')  # print the how many images contained in the training set
print('The shape of the image is', train_images[0].shape)  # print the shape of the image



##_--------------Data Preprocessing ----------------------------------##
#Goal of this section: segregating the data into small chunks, changing the label, seperate train & test data

num_samples=10000 #select the size of data for segregating

train_images_section= train_images[0:num_samples]
train_labels_section= train_labels[0:num_samples]

train_images_section=train_images_section/255    # data Normalization
#print('one instance of train image is :',train_images_section[0])

for i in range(0,num_samples): 
 if(train_labels_section[i]==3):
   train_labels_section[i]=1
 else:
   train_labels_section[i]=0

 
X_train=train_images_section
y_train= train_labels_section

print('The training set contains', len(X_train), 'images')  # print the how many images contained in the training set
print('The shape of the train image is', X_train[0].shape)  # print the shape of the image



##------------Preparing vectorized form of data--------------------##
a={}
m=X_train.shape[0]
print('m is:',m)
print(X_train.shape);
a[0]=X_train.reshape(X_train.shape[1]*X_train.shape[2],X_train.shape[0])
print('shape of x train_m:',a[0].shape)
y_train_v=y_train.reshape(y_train.shape[0],1).T
print('shape of y_train_v:',y_train_v.shape)

alpha= 0.00002

#-----------------------------9999999999999999999999999999-------------------------

modelstruct={}
def layer(layernumber,numberofUnits,Activation):
 global modelstruct
 layerdata={}
 layerdata["numU"]=numberofUnits
 layerdata["actF"]=Activation
 #print(layerdata)
 layer=('Layer'+str(layernumber))
 modelstruct[layer]=layerdata
 #print(modelstruct)


def createmodelparameter():
  global modelstruct
  l=[]
  print(modelstruct)
  print(len(modelstruct))
  for x in modelstruct:
   l.append(modelstruct[x]["numU"])
  l=np.array(l)
  return l

def designNrun():
 global a
 layer(0,784,'None')
 layer(1,5,'relu')
 layer(2,1,'sigmoid')
 cost=[]
 iteration=[]
 w_size_list=createmodelparameter()
 print('Weight size list:',w_size_list)
 w={};
 z={};
 dz={}
 dw={}
 db={}
 b=[]
 bias={}
 for i in range(len(w_size_list)-1):
     w[i+1]= np.random.randn(w_size_list[i+1],w_size_list[i])
     print('Shape of weight[',i+1,']:',w[i+1].shape)  
     b.append(0)
     bias[i+1]=0
     bias[i+1]=np.array(bias[i+1])
     
 #print('Bias:',bias)
 print('Number of active layers:',len(b))

 activ=[]
 for x in modelstruct:
     activ.append(modelstruct[x]["actF"])
  
 print('Activation type of layers:',activ)
 print('Size of a[0] is:',a[0].shape)
 
    
 for itr in range(1000): 
  
#Forward prop::
  for i in range(len(b)):
     
     print('--------------layer',i+1,' forward prop-----------------------')
     
     z[i+1]=np.dot(w[i+1],a[i])+bias[i+1]
     if activ[i+1]=='relu':
       a[i+1]=relu(z[i+1])
       print('i am in :',activ[i+1])
     elif activ[i+1]=='sigmoid':
       a[i+1]=sigmoid(z[i+1])
       print('i am in :',activ[i+1])
     
     print('size of w[',i+1,']: ',w[i+1].shape)
     print('size of z[',i+1,']: ',z[i+1].shape)
     print('size of a[',i+1,']: ',a[i+1].shape)
     print('size of bias[',i+1,']:',bias[i+1].shape)

#calculation of cost J :
     if(i+1==len(b)):    
      J=-y_train_v*(np.log(a[i+1]))-(1-y_train_v)*np.log(1-a[i+1]) 
      J= np.sum(J)/m
      cost.append(J)
      iteration.append(itr)
      print('shape of a[L]:',a[i+1].shape)
     
#Backward Prop::
  back= len(b)
  print('++++++++++++++++++++======================+++++++++++++++++++++++++++')
  for j in range(len(b)):
   
    if(back-j==len(b)):
       print('--------------layer',back-j,'Back prop-------------------------')
       
       dz[back-j]=sigmoidD(a[back-j],y_train_v)
       dw[back-j]= (1/m)* np.dot(dz[back-j],a[back-j-1].T)
       db[back-j]=(1/m)*np.sum(dz[back-j],axis=1,keepdims=True)
       
       print('size of a[',back-j,']:',a[back-j].shape)
       print('size of y_train_v:',y_train_v.shape)
       print('size of dz[',back-j,']:',dz[back-j].shape)    
       print('size of a[',back-j-1,']:',a[back-j-1].shape)
       print('size of dw[',back-j,']:',dw[back-j].shape)
       print('size of db[',back-j,']:',db[back-j].shape)
       
       
    else :
      print('--------------layer',back-j,'Back prop-------------------------')
      
      if(activ[back-j]=='relu'): 
       dz[back-j]=np.dot(w[back-j+1].T,dz[back-j+1])* reluD(z[back-j])
       dw[back-j]= (1/m)* np.dot(dz[back-j],a[back-j-1].T)
       db[back-j]=(1/m)*np.sum(dz[back-j],axis=1,keepdims=True)
       
       
       print('j is :',back-j)
       print('size of dz[',back-j,']:',dz[back-j].shape)
       print('size of dw[',back-j,']:',dw[back-j].shape)
       print('size of db[',back-j,']:',db[back-j].shape)    
       print('size of w[',back-j+1,']:',w[back-j+1].shape)
       print('size of dz[',back-j+1,']:',dz[back-j+1].shape)
       print('size of z[',back-j,']:',z[back-j].shape)
       print('size of a[',back-j-1,']:',a[back-j-1].shape)
       
     #if(activ[back-j]=='sigmoid'): 
     # dz[back-j]=np.dot(w[back-j+1].T,dz[back-j+1])* sigmoidD(z[back-j])
     # dw[back-j]= (1/m)* np.dot(dz[back-j],a[back-j-1].T)
     # db[back-j]=(1/m)*np.sum(dz[back-j],axis=1,keepdims=True)
     # print('j is :',back-j) 
 

#value updation ::
  print('++++++++++++++++++++======================+++++++++++++++++++++++++++')
  for k in range(len(b)):
      print('--------------Layer:',k+1)
      print('size of w:',w[k+1].shape)
      print('size of dw:',dw[k+1].shape)
      w[k+1]=w[k+1]-alpha*dw[k+1]
      print('size of bias before [',k+1,']:',bias[k+1].shape)
      print('size of db[',k+1,']:',db[k+1].shape)
      bias[k+1]=bias[k+1]-alpha*db[k+1]
      print('size of bias after [',k+1,']:',bias[k+1].shape)
      
      
 pyplot.title("cost function")
 pyplot.xlabel("iteration")
 pyplot.ylabel("cost");
 pyplot.plot(iteration,cost, color='red',linewidth=2,label="J cost") 

def relu(z):
    return np.maximum(z,[0])
def reluD(z):
    return ((z>0)*1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoidD(a,y):
    return (a-y)