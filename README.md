# Build Regularized Linear Regression From Scratch
input parameter:
1. data point (x, y)
```
-5.0,51.76405234596766
-4.795918367346939,45.42306433039972
-4.591836734693878,41.274448104888755
```
2. the number of polynomial bases n
3. lambda (only for LSE)

Function:
1. LSE
2. Newton's method in optimization
3. Visualization (matplotlib)

Output:
n = 2, lambda = 0
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/1.LSE.jpg)
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/1.Newton.jpg)
n = 3, lambda = 0
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/2.LSE.jpg)
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/2.Newton.jpg)
n = 3, lambda = 10000
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/3.LSE.jpg)
![image](https://github.com/ginagigo123/Regularized-Linear-Regression-From-Scratch/blob/main/result/3.Newton.jpg)


## Build Matrix From Scratch
因為作業要求不能使用套件，所以需要自己自建一個Matrix class，也理所當然不能使用numpy了