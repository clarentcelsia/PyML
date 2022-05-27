# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:11:00 2022

@author: user
"""

import numpy as np
#1. 
    #a. Create new matrix
    #b. do summation, subtraction, multiplication of the given matrix
    #c. Find determinant of the given matrix
#a
size = input('size of list: ')
row = input('length of row: ')
col = input('length of column: ')

arrs = []
rnums = []
cnums = []
for i in range(int(size)):
    for r in range(int(row)):
        for c in range(int(col)):
            num = input('input number for row-%d col-%d ' %(r,c))
            cnums.append(int(num))
        if c == (int(col)-1):
            rnums.append(cnums.copy())
            cnums.clear()
    if r == (int(row)-1):
        arrs.append(rnums.copy())
        rnums.clear()
        
print(arrs)

#b&c
sums = sub = mul = np.array([])
dets = []
for lists in arrs:
  print(lists)
  if len(sums)==0:
    sums = sub = mul = np.array(lists)
  else:
    sums = sums + (np.array(lists))
    sub = sub - (np.array(lists))
    mul = mul.dot((np.array(lists)))
    dets.append(np.linalg.det(lists))

print(sums)
print(sub)
print(mul)
print(dets)

#2. Find x, y, z
#x + 3y + 5z = 13
#7x + 12y + 21z = 123
#5x + 18y + 3z = 51

a = [[1, 3, 5], [7, 12, 21], [5, 18, 3]]
b = [13, 123, 51]

np.linalg.solve(a, b)
