# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:41:43 2021

@author: GINATSAI
content: Instead of using Numpy, build Matrix module From scratch.
         +, -, *, @, Transpose
"""

class Matrix:
    # Constructor
    def __init__(self, dims, fill):
        self.row = dims[0]
        self.col = dims[1]
        # constant
        if isinstance (fill, (int, float)):
            self.A = [ [fill] * self.col for i in range(self.row)]
        # 2d array
        elif isinstance (fill, list):
            self.A = fill
    
    # print Matrix
    def __str__(self):
        mtxStr = ''
        for r in range(self.row):
            for c in range(self.col):
                mtxStr += str(self.A[r][c])
                if c != self.col - 1:
                    mtxStr += ', '
            mtxStr += '\n'
        return mtxStr
        
    # A + B or A + 2
    def __add__(self, other):
        # new Matrix
        C = Matrix(dims = (self.row, self.col), fill = 0)
        
        if isinstance (other, Matrix):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] = self.A[i][j] + other.A[i][j]
        
        elif isinstance (other, (int, float)):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] += other
        
        return C
    
    # right-hand-side
    def __radd__(self, other):
        return self.__add__(other)
    
    # A - B or A - 2
    def __sub__(self, other):
        # new Matrix
        C = Matrix(dims = (self.row, self.col), fill = 0)
        
        if isinstance (other, Matrix):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] = self.A[i][j] - other.A[i][j]
        
        elif isinstance (other, (int, float)):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] -= other
        
        return C
    
    # A * 2 or A * b matrix-matrix pointwise multiplication
    def __mul__(self, other):
        C = Matrix(dims = (self.row, self.col), fill = 1)
        
        # Scalar multiplication
        if isinstance(other, (int, float)):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] = self.A[i][j] * other
        elif isinstance(other, Matrix):
            for i in range(self.row):
                for j in range(self.col):
                    C.A[i][j] = self.A[i][j] * other.A[i][j]
        return C
                    
        
    # 2.0 * A
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # A @ B matrix-matrix multiplication
    def __matmul__(self, other):
        if isinstance (other, Matrix):
            C = Matrix(dims = (self.row, other.col), fill = 0)
            if self.col != other.row:
                print("InValid multiplication")
                return
            
            for i in range(self.row):
                for j in range(other.col):
                    tmp = 0
                    for k in range(self.col):
                        tmp = tmp + self.A[i][k] * other.A[k][j]
                    C.A[i][j] = tmp
        return C
    
    # Matrix Transpose
    def T(self):
        C = Matrix(dims = (self.col, self.row), fill = 0)
        for i in range(self.row):
            for j in range(self.col):
                C.A[j][i] = self.A[i][j]
        return C
    
    # get the value of Matrix[i, j]
    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key[0], key[1]
            return self.A[i][j]
    
    # set the value of Matrix[i, j]
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key[0], key[1]
            self.A[i][j] = value
            
                    

if __name__ == '__main__':
    # test
    A = Matrix(dims = (3,2), fill = 1)
    
    B = Matrix(dims = (2,6), fill = 6.6)
    C = A @ B
    print(C)
    tmp = C.T()
    print(tmp)
    tmp = tmp @ C
    print(C.T() @ C)
    D = Matrix(dims = (3, 2), fill = [[1, 2], [3, 4], [5, 6]])
    print(D)
    G = Matrix(dims = (2, 2), fill = 2)
    print(D @ G)
    A[0,0]
    A[0,0] = 2
    print(A)