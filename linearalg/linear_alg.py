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


u = np.array([1,2,3])
v = np.array([2,1,2])

#VECTOR PROJECTION
#https://www.omnicalculator.com/math/vector-projection#:~:text=Here%20is%20the%20vector%20projection,also%20called%20the%20scalar%20product.
#p = (a·b / b·b) * b
#finding norm of the vector v
v_norm = np.sqrt(sum(v**2))
proj_u_on_V = (np.dot(u,v)/v_norm**2)*v
print('projection of vector u on vector v is: ', proj_u_on_V)

#BASIS
#https://www.geeksforgeeks.org/basis-vectors-in-linear-algebra-ml/
vec = [3,4]
#standard
#w = bobot
w1 = [-2,4]
w2 = [2,1]
#New basis
new_vec = np.linalg.inv(np.array([w2,w1])).dot(vec)
print(new_vec)

