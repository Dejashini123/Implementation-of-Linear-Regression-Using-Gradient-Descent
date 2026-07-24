# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the dataset and select the input (R&D Spend) and output (Profit) variables.
3. Initialize the slope, intercept, learning rate, and number of iterations.
4. Apply Gradient Descent to update the slope and intercept until convergence.
5. Display the regression line and predicted output.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MAALINI B N
RegisterNumber: 212224060136
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("exp_3_50_Startups.csv")

X = data["R&D Spend"].values
Y = data["Profit"].values

X = (X - X.mean()) / X.std()

m = 0
c = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):

    y_pred = m * X + c

    dm = (-2/n) * np.sum(X * (Y - y_pred))
    dc = (-2/n) * np.sum(Y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Slope (m):", m)
print("Intercept (c):", c)

y_pred = m * X + c

plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("R&D Spend (Scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
```

## Output:

<img width="842" height="636" alt="image" src="https://github.com/user-attachments/assets/3f5a267f-a969-4867-9a73-ddffbdf5a9ea" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
