# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:16:37 2021

@author: GINATSAI
content: 
    Regularized linear model regression (polynomial basis) & visualization
    1. LSE : use LU decomposition to find the inverse of A.T() @ A + lamda * I
    2. Newton's method
"""
from Matrix import Matrix
import matplotlib.pyplot as plt
import time # for calculating the execution time

def IdentityMatrix(n):
    I = Matrix(dims = (n, n), fill = 0)
    for i in range(n):
        I[i, i] = 1
    return I

def findLU(n, matrixA):
    # Find the L U for the decomposition
    L = Matrix(dims = (n, n), fill = 0)
    U = Matrix(dims = (n, n), fill = 0)
    
    # Fill the value of L
    for i in range(n):
        L[i, i] = 1
    
    # Calculate for the lower, upper matrix, i for row index, j for col index
    for i in range(n):
        for j in range(n):
            tmp = matrixA[i, j] 
            for k in range(n):
                if j != k:
                    tmp = tmp -  L[i, k] * U[k, j]

            # lower matrix     
            if i > j :
                if U[j , j] != 0:
                    L[i, j] = tmp / U[j, j]
                else :
                    L[i, j] = tmp
            # Upper matrix
            else:
                if L[j, j] != 0:
                    U[i, j] = tmp / L[j, j]  
                else:
                    U[i, j] = tmp
    return L, U

def inverse(n, matrixA):
    # Ax = I => LUx = I
    # Find the L U for the decomposition
    L, U = findLU(n, matrixA)
                    
    # LZ = c = I -> c using forward substitution
    C = IdentityMatrix(n)
    Z = Matrix(dims = (n, n), fill = 0)
    for i in range(n):
        for j in range(n):
            tmp = C[i, j]
            #text = str(i) + ', ' + str(j) + ' : ' + str(C[i, j])
            for k in range(n):
                if k!= i:
                    tmp -= L[i, k] * Z[k, j]
                    #text += ' - ' + str(L[i, k] * Z[k, j])
            if L[i, i] != 0:
                Z[i, j] = tmp / L[i, i]
            else:
                Z[i, j] = tmp
            #text += ' = ' + str(tmp) + ' / ' + str(L[i, i])
            #print(text)
            
    #print(L @ Z) # should be indentity matrix
    #print(Z)
    
    # Ux = Z using backward substitution
    X = Matrix(dims = (n, n), fill = 0)
    for i in range(n-1, -1, -1):
        for j in range(n):
            tmp = Z[i, j]
            #text = str(i) + ', ' + str(j) + ' : ' + str(Z[i, j])
            for k in range(n):
                if k!= i:
                    tmp -= U[i, k] * X[k, j]
                    #text += ' - ' + str(U[i, k] * X[k, j])
            if U[i, i] != 0:
                X[i, j] = tmp / U[i, i]
            else:
                X[i, j] = tmp
            #text += ' = ' + str(tmp) + ' / ' + str(U[i, i]) + ' = ' + str(X[i, j])
            #print(text)
            
    #print(U @ X) # should be Z
    #print(Z)
    #print(result @ X) # near the identity matrix
    #print(X)
    
    return X
    
def fillMatrix(x, y, base_n):
    # create computing matrix
    matrixA = []
    matrixb = []
    for data in x:
        tmp = []
        for j in range(base_n - 1, -1, -1):
            tmp.append(pow(data, j))
        matrixA.append(tmp)
    for data in y:
        matrixb.append([data])
    
    A = Matrix(dims = (len(x), base_n), fill = matrixA)
    b = Matrix(dims = (len(y), 1), fill = matrixb)
    return A, b
                    
def LSE(A, b, lamd):
    # Identity matrix
    I = IdentityMatrix(n)
    
    # A.Tranpose @ A + lamda(scalar) * I
    x = A.T() @ A + lamd * I
    x_inv = inverse(x.row, x);
    x = x_inv @ A.T() @ b
    return x

def Newton(A, b, n):
    # initialize the x0 to whatever
    x0 = []
    for i in range(n):
        x0.append([0])
    x0 = Matrix(dims = (n, 1), fill = x0)
    
    value = 1
    times = 0
    while value > 0.001:
    
        deltaF = 2 * A.T() @ A @ x0 - 2 * A.T() @ b
        HessionF = 2 * A.T() @ A
        
        x1 = x0 - inverse(HessionF.row, HessionF) @ deltaF
        
        # calculate x[i] - x[i-1] value
        tmp = x1 - x0
        value = 0
        for i in range(n):
            value += tmp[i, 0]
        x0 = x1
        times += 1
    
    print("Newton's method 迭代次數:", times - 1)
    return x1

def printLine(x):
    output = ""
    for i in range(x.row - 1):
        output = output + str(x[i, 0]) + "x^" + str(x.row - 1 - i) + " + "
    output = output + str(x[ x.row - 1, 0])
    print("Fitting ine : ", output)

def computeError(A, X, b):
    predictY = A @ X
    error = predictY - b
    
    errorSum = 0
    for i in range(b.row):
        errorSum += error[i, 0] * error[i, 0]
    print("Total error : ", errorSum)
    
def plotline(x, y, line, n):
    predictY = []
    # i for number of y
    for i in range(len(x)):
        tmp = 0
        # j for power
        for j in range(n):
            tmp += line[j, 0] * x[i] ** (n - 1 - j)
        predictY.append(tmp)
    #print(predictY)
    
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axvline(color = 'k') # black
    plt.axhline(color = 'k')
    
    plt.scatter(x, y, color = 'r', edgecolors = 'k') # red
    plt.plot(x, predictY, color = 'b', linewidth = 2.0) # blue    
    plt.show()
    
# read file
f = open('testfile.txt', 'r')
lines = f.readlines()
x = []
y = []
for line in lines:
    text = line.split(',')
    x.append(float(text[0]))
    y.append(float(text[1]))

# LSE
n = int(input("plz enter the number of base : "))
lamb = float(input("plz enter the lambda : "))
A, b = fillMatrix(x, y, n);

print("\n---------- LSE -----------")
start_time = time.clock()
line1 = LSE(A, b, lamb)
print("LSE takes ",time.clock() - start_time, "s")
printLine(line1)
computeError(A, line1, b)
plotline(x, y, line1, n)

# Newton's method in optimization
print("---------- Newton's method -----------")
start_time = time.clock()
line2 = Newton(A, b, n)
print("Newton's method takes", time.clock() - start_time, "s")
computeError(A, line2, b)
plotline(x, y, line2, n)
