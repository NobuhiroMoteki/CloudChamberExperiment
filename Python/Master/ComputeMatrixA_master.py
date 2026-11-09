# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:14:31 2022

@author: Moteki
"""
import numpy as np
fname= 'simulated_XY_dataset.csv'
dataset = np.loadtxt(fname, delimiter=',', skiprows=1)
X= dataset.transpose()[0:2,:]
Y= dataset.transpose()[2:4,:]
Y= np.insert(Y,0,1,axis=0)
A= np.matmul(np.matmul(X,Y.transpose()),np.linalg.inv(np.matmul(Y,Y.transpose())))
print(A)
