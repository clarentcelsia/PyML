# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:47:21 2022

@author: Allen
"""

# https://www.geeksforgeeks.org/program-for-rank-of-matrix/
class rankMatrix(object):
    def __init__(self, Matrix):
        self.R = len(Matrix)
        self.C = len(Matrix[0])
         
    # Function for exchanging two rows of a matrix
    def swap(self, Matrix, row1, row2, col):
        for i in range(col):
            temp = Matrix[row1][i]
            Matrix[row1][i] = Matrix[row2][i]
            Matrix[row2][i] = temp
             
    # Function to Display a matrix
    def Display(self, Matrix, row, col):
        for i in range(row):
            for j in range(col):
                print (" " + str(Matrix[i][j]))
            print ('\n')
             
    # Find rank of a matrix
    def rankOfMatrix(self, Matrix):
        rank = self.C
        for row in range(0, rank, 1):
             
            # Before we visit current row
            # 'row', we make sure that
            # mat[row][0],....mat[row][row-1]
            # are 0.
     
            # Diagonal element is not zero
            if Matrix[row][row] != 0:
                for col in range(0, self.R, 1):
                    if col != row:
                         
                        # This makes all entries of current
                        # column as 0 except entry 'mat[row][row]'
                        multiplier = (Matrix[col][row] /
                                      Matrix[row][row])
                        for i in range(rank):
                            Matrix[col][i] -= (multiplier *
                                               Matrix[row][i])
                                                 
            # Diagonal element is already zero.
            # Two cases arise:
            # 1) If there is a row below it
            # with non-zero entry, then swap
            # this row with that row and process
            # that row
            # 2) If all elements in current
            # column below mat[r][row] are 0,
            # then remove this column by
            # swapping it with last column and
            # reducing number of columns by 1.
            else:
                reduce = True
                 
                # Find the non-zero element
                # in current column
                for i in range(row + 1, self.R, 1):
                     
                    # Swap the row with non-zero
                    # element with this row.
                    if Matrix[i][row] != 0:
                        self.swap(Matrix, row, i, rank)
                        reduce = False
                        break
                         
                # If we did not find any row with
                # non-zero element in current
                # column, then all values in
                # this column are 0.
                if reduce:
                     
                    # Reduce number of columns
                    rank -= 1
                     
                    # copy the last column here
                    for i in range(0, self.R, 1):
                        Matrix[i][row] = Matrix[i][rank]
                         
                # process this row again
                row -= 1
                 
        # self.Display(Matrix, self.R,self.C)
        return (rank)

# This code is contributed by Vikas Chitturi

#-------------------------------------------------
import numpy as np

#Given vector A as following:
A = [[3],
     [4]]

#Find unit vector of A matrix
  # U_Vec_A = A/|A|
  # |A| = magnitude -> besaran scalar = sqrt(3^2 + 4^2)
  #nb. 
    # Unit vector yang men-define arah in coordinate system
    # Vector merupakan product dari Unit Vector . Scalar (|A|)
magnitude_A = np.sqrt(np.power(A[0],2)) + np.power(A[1],2)
unit_vector_A = A/magnitude_A
print(unit_vector_A)

#===================================================================
U = [[1, -2, 4]] #1->(-2), 1->(4), dst
V = [[2, 5, 2]]
#ORTHOGONAL VECTOR
  # Requirement for 2 n-dimensional vectors are orthogonal to each other:
    # Their dot product(scalar_product) = 0
        # A.B = sigma(n,i=1) Ri*Ci = A-transpose.B
dot_U_V = np.dot(V, np.transpose(U))
print(dot_U_V)

#ORTHONORMAL VECTOR
  # a set of vectors S is orthonormal if every vector in S has magnitude = 1 and 
  # the set of vectors are mutually orthogonal.
# Normalize the matrix = unit_vector
magnitude_U = np.sqrt((np.power(U[0][0],2) + np.power(U[0][1],2) + np.power(U[0][2],2))) 
magnitude_V = np.sqrt((np.power(V[0][0],2) + np.power(V[0][1],2) + np.power(V[0][2],2)))
unit_vec_U = np.transpose(U)/magnitude_U
unit_vec_V = np.transpose(V)/magnitude_V
# Find the magnitude of unit_vector = unit_magnitude
unit_mag_U = np.sqrt((np.power(unit_vec_U[0][0],2) + np.power(unit_vec_U[1][0],2) + np.power(unit_vec_U[2][0],2))) # >>> 1
unit_mag_V = np.sqrt((np.power(unit_vec_V[0][0],2) + np.power(unit_vec_V[1][0],2) + np.power(unit_vec_V[2][0],2))) # >>> 1
print(unit_mag_U)
print(unit_mag_V)

#BASIS VECTOR
  # Basis vector means vector that linearly independent of each other and must span the whole space (linear combination).
  # i.e Example of Basis vector as following:
v1 = [[1],
      [0]]
v2 = [[0],
      [1]]
# When (basis) vector v1 multiplied by any scalar, it doesnt give the same result as vector v2
scalar = 2
v1_result = scalar*np.array(v1) # >>> [[2],[0]]
v2_result = scalar*np.array(v2) # >>> [[0],[2]]

# By multiplying it with basis vector can give the same value with that desired vector. (linear combination)
any_vector = [[2],[1]]
scalar1 = 2
scalar2 = 1
result = scalar1*np.array(v1) + scalar2*np.array(v2) # >>> [[2],[1]]

# Basis vector is not unique.
# Try another basis vector [[1],[1]] & [[1],[-1]] -> Linearly independent & can be written as linear combination
# Find Basis Vector
R4 = [ [[6],[5],[8],[11]], [[1],[2],[3],[4]], [[9],[4],[7],[10]], [[2],[2],[1],[4]] ]
Matrix = [[6,1,9,2],[5,2,4,2],[8,3,7,1],[11,4,10,4]]
RankMatrix = rankMatrix(Matrix)
rank = RankMatrix.rankOfMatrix(Matrix)
# If the rank of the matrix is 1 then we have only 1 basis vector, if the rank is 2 then there are 2 basis vectors and so on..
# https://www.geeksforgeeks.org/basis-vectors-in-linear-algebra-ml/?ref=rp
# we can pick any 2 linearly independent columns here and then those could be the basis vectors. 
# So, the basis vector randomly choose v1 = [[6],[5],[8],[11]], v2 = [[1],[2],[3],[4]]

# Now, We can store the basis vector only to reconstruct (using linear combination) the data without storing the whole data number
# because they're linearly independent.

# EIGENVECTOR & EIGENVALUES 
    # (Ax = Ex), where x is eigenvector and E = eigenvalue, to check if the given E is true 
    # used for reducing the linear operation.
    
import numpy as np
from numpy.linalg import eig

a = np.array([[0, 2], 
              [2, 3]])
w,v=eig(a)
print('E-value:', w)
print('E-vector', v)

# EIGENVALUES (|A-EI| = 0), where E = eigenvalue and I = identity matrix
# a set of scalar
# EIGENVECTOR
# a set of vector