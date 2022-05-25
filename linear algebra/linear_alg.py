# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:55:00 2022

@author: Allen
"""

import numpy as np
import matplotlib.pyplot as plt

#MATRICES
a = [[1,2,3],
     [2,3,1],
     [3,2,1]]
print(a)
print(a[0][1], end="\n")

#Collect only 3rd col
column = []
for numbers in a:
    print(numbers) #list
    column.append(numbers[2])
print(column)

b = [[1,2,5],
     [2,4,2],
     [2,5,3]]

#Numpy
#https://docs.scipy.org/doc/numpy-1.15.1/user/quickstart.html
mA = np.array(a)
mB = np.array(b)
print(mA + mB)
print(mA * mB) #Array multiplication instead of matrix multiplication 
print(mA.dot(mB))
print(mA.transpose())
print(np.linalg.inv(mA))
print(mB.shape)

#Numpy reshape
reshape = np.arange(15, dtype=np.int64).reshape(3,5)
print(reshape)
print('row with index 0 only: %s' %(reshape[:1]))
print('2 even indexed rows only: %s' %(reshape[::2]))
print('1st column: %s' %(reshape[:,0])) #[r,c]
print('2 element of first row with 2 element of 2nd row: %s' %(reshape[:2, :2]))
print('(r22,r23)-(r32,r33): %s' %(reshape[1:, 1:3]))
print('(r22,r25)-(r32,r35): %s' %(reshape[1:, 1::3]))
    
#nb: 
    # : -> whole element
    # ::n -> extended slices
    # as its name, it only takes the index of the element with n sequence/step.
    # i.e '0,1,2,3,4' ::3 -> element with index 0,3,6,9,..