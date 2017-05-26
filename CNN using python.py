# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import pandas as pd
import lasagne
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Loading the dataset
train=pd.read_csv('C:/Users/Jyoti Prakash/Desktop/Digit Recognition/train.csv')
Y_train=train.iloc[1:25000,0]
Y_val=train.iloc[25000:35000,0]
Y_test=train.iloc[35000:42000,0]
X_train=train.iloc[1:25000,1:785]
X_val=train.iloc[25000:35000,1:785]
X_test=train.iloc[35000:42000,1:785]

X_train = X_train.reshape(( 1, 28, 28))
