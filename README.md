# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Load necessary libraries for data handling, metrics, and visualization.

2. **Load Data**: Read the dataset using `pd.read_csv()` and display basic information.

3. **Initialize Parameters**: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. **Gradient Descent**: Perform iterations to update `m` and `c` using gradient descent.

5. **Plot Error**: Visualize the error over iterations to monitor convergence of the model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Tharika S
RegisterNumber: 212222230159
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![image](https://github.com/user-attachments/assets/777ab209-084d-4326-9b67-89874859a340)
```
dataset.info()
```
## Output:
![image](https://github.com/user-attachments/assets/b8a4f4ef-7d00-41b2-9d93-f4b0681f0075)
```
x=dataset.iloc[:,:-1].values
print(x)
y =dataset.iloc[:,-1].values
print(y)
```
## Output:
![image](https://github.com/user-attachments/assets/a60731d6-90e3-485f-b4be-1e664a8e7d7d)
```
x.shape
```
## Output:
![image](https://github.com/user-attachments/assets/5e5f3f5d-29fe-4ec3-9841-c6ff23025393)
```
y.shape
```
## Output:
![image](https://github.com/user-attachments/assets/56ecf270-c26c-4348-a6ce-200749cecd06)
```
m=0
c=0
L=0.001
epochs=5000
n=float(len(x))
error=[]
for i in range(epochs):
  y_pred=m*x+c
  d_m=(-2/n)*sum(x*(y-y_pred))
  d_c=(-2/n)*sum(y-y_pred)
  m=m-L*d_m
  c=c-L*d_c
  error.append(sum(y-y_pred)**2)
print(m,c)
```
## Output:
![image](https://github.com/user-attachments/assets/de72a9f3-35e7-4f0a-9fcb-97d4f6bc4929)
```
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
```
## Output:
![image](https://github.com/user-attachments/assets/f4c01b8f-1035-4682-a542-e039aa5ff05c)
![image](https://github.com/user-attachments/assets/76b8ba61-4003-4bec-b6ac-ed63c33864ca)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
