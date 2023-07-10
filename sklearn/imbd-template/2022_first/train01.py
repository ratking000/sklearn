import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error

# Read the CSV file
df = pd.read_csv(r'C:\Users\User\sklearn\sklearn\imbd-template\2022_first\2022-train-v2.csv')

# Erase the columns with missing values
df = df.dropna(axis=1)

y = df.iloc[:, 0]
# print(df.shape)
X = df.iloc[:, 6:]  

# Split the data into trainning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2023)

# Use Linear Regression to train the model and test the model
model = linear_model.LassoLars()
# model = RandomForestRegressor()

# model = make_pipeline(PolynomialFeatures(4), LinearRegression())

print("Params:", model.get_params())
params = [
    {'alpha':[0.1, 0.3, 0.5, 0.7, 0.9]}
]

# best_model = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse:", rmse)