# -*- coding: utf-8 -*-
"""
Created on Fri May 27 23:00:03 2022

@author: Allen
"""

# TRANSFORMATION MATRIX
    # A matrix that transforms one vector into another vector.
    # Rotation, Translation, Reflection, Shearing, Scaling

import matplotlib.pyplot as plt
import numpy as np
import string

xs = []
ys = []
color = 'rgbr'

def draw(triangle):
  fig = plt.figure(0, figsize=(5,3))
  plt.title('normal')
  # plt.gca() # get current axis, default (1,1)

  for i, row in enumerate(triangle):
    d_prod = row @ I # recall: basis*, >>> [1.0 1.0], [5.0 1.0], ...
    x, y = d_prod # >>> x = 1, 5, 1 & y = 1, 1, 4

    xs.append(x) # >>> [1,5,1]
    ys.append(y) # >>> [1,1,4]

    # with scatter, you have more control over the points
    # meanwhile plot is when you want to create a line between 1 point to others
    plt.scatter(x, y, c=color[i])
    plt.text(x + 0.1, y + 0.05, f"{string.ascii_letters[i]}")

  xs.append(xs.copy()[0])
  ys.append(ys.copy()[0])

  plt.plot(xs, ys, color="gray", linestyle='dotted') # >>> (1,0)->(5,0)->(1,4)->(1,0))

  xs.clear()
  ys.clear()

def build_reflection(reflection, i):
  fig = plt.figure(1)
  plt.title('reflection(x-axis, y-axis)')

  x = reflection[0]
  y = reflection[1]

  # for i, item in enumerate(x):
  #   plt.scatter(item, y[i], c=color[i])
  
  x = np.append(x, x[0])
  y = np.append(y, y[0])

  # 1: nrows, 2: ncols, 1: current_index
  plt.subplot(2,1,i)
  plt.plot(x, y, color="gray", linestyle='dotted')

def reflection(triangle):
  # reflection based on x-axis & y-axis
  xreflection = np.array([[1,0],[0,-1]])
  yreflection = np.array([[-1,0],[0,1]])

  xymatrix = [] # >>> [[1,5,1],[1,1,4]]
  for i, row in enumerate(triangle):
    d_prod = row @ I # recall: basis*, >>> [1.0 1.0], [5.0 1.0], ...
    x, y = d_prod # >>> x = 1, 5, 1 & y = 1, 1, 4

    xs.append(x) # >>> [1,5,1]
    ys.append(y) # >>> [1,1,4]

  xymatrix.append(xs.copy())
  xymatrix.append(ys.copy())

  # find the new matrix for creating new reflected triangle
  # across the x-axis
  xreflection_matrix = xreflection @ xymatrix # >>> [[ 1.  5.  1.],[-1. -1. -4.]]
  build_reflection(xreflection_matrix, 1)

  # across the y-axis
  yreflection_matrix = yreflection @ xymatrix # >>> [[-1. -5. -1.],[ 1.  1.  4.]]
  build_reflection(yreflection_matrix, 2)

if __name__=="__main__":
  # 2x2 Identity Matrix
  I = np.eye(2)
  # Points for creating r-triangle shape
  A, B, C = (1,1), (5,1), (1,4)

  triangle = np.array([A, B, C])

  draw(triangle)
  reflection(triangle)

