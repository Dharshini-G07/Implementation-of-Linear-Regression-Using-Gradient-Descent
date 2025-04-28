# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Dataset: Read the dataset (50_Startups.csv) into a pandas DataFrame.

2.Extract Features and Target: Separate the dataset into features (X) and target variable (y).

3.Convert Data to Numeric: Convert the feature data to float type if necessary.

4.Scale Features: Scale the feature data (X) using StandardScaler to standardize the values.

5.Scale Target (Optional): Optionally scale the target variable (y) for better performance in gradient descent.

6.Initialize Parameters: Initialize the model parameters (theta) with zeros.

7.Gradient Descent: Apply gradient descent to minimize the cost function and update the parameters (theta).

8.Train the Model: Train the model using the scaled features and target variable.

9.Make Predictions: Predict the output using the trained model for new data.

10.Inverse Scaling: (If target was scaled) Apply inverse scaling to the prediction to get it back to the original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Priyadharshini G
RegisterNumber:  212224230209
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
## data.head()
![image](https://github.com/user-attachments/assets/1ff649af-c10e-4a50-9892-3221e1202e71)
## X
![image](https://github.com/user-attachments/assets/fa0755f2-5362-4fe6-8d01-bf93d5615bcf)
## X1_scaled
![image](https://github.com/user-attachments/assets/83527c1c-2106-46fe-80cc-ee018a1c0174)
## Prediction
![image](https://github.com/user-attachments/assets/5f49a753-054f-4d37-bf4d-ab008cf34098)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
