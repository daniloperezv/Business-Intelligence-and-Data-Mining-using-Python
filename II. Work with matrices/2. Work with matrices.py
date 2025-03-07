##WORK WITH MATRICES##
######################

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:52:04 2025

@author: danilo
"""

#We define a list called Y
Y=[1,2,3,4,5]

#We define a list called X
X=[0.1,0.4,0.66,0.88,0.99]

#Now, we import the numpy library
import numpy as np

#Using numpy, we transform X and Y in matrices
mat_Y=np.array(Y) #First, we create it without dimensions
mat_Y=mat_Y.reshape(-1,1) #Then, with reshape we assign a dimension (-1 means many rows and 1 means 1 column)

mat_X=np.array(X)
mat_X=mat_X.reshape(-1,1)

#We can transpose the original matrices 
X_transposed=np.transpose(mat_X)

x_transposed2=mat_X.T

#We can create an identity matrix
ones=np.ones((100,10))
identity=np.eye(5)

#Dot product
X_mult_ident=np.dot(X_transposed, identity)

#Concatenate matrices
X_c=np.concatenate((identity,mat_X),axis=1)  #the command axis=1 concatenate columns, while axis=0 concatenate rows

#Create a matrix full of random numbers
mat_nums=np.arange(30).reshape(3,10)

mat_rand=np.random.rand(5,5)
mat_rand_int=np.random.randint(0,100,(5,5))
mat_rand_normal=np.random.normal(0,1,(5,5))


#We can find an inverse of a matrix with numpy
from numpy.linalg import inv
M_X_inv = inv(mat_rand_normal)

