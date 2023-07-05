import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Read the CSV file
df = pd.read_csv('/home/joey/program/sklearn/imbd-template/2022_first/2022-train-v2.csv')

# Step 2: Summary of the dataframe
# The describe function only shows some fields, I want it to display all fields
# pd.set_option('display.max_columns', None)

# summary = df.describe()
# print(summary)

# Identify the missing fields, and sort them
# print(df.isnull())

# print(df.isnull().sum().sort_values(ascending=False))
# print(df.isnull().sum().sum())

# Erase the columns with missing values
df = df.dropna(axis=1)

# Set y as the target variable, which is the first column of the dataframe
y = df.iloc[:, 4]
# print(y)
# print(df.shape)

# Set X as the feature variables, which are the remaining columns of the dataframe except the first 6 columns
X = df.iloc[:, 6:]

# column_names = df.columns.tolist()

# x_column = column_names[0]


# for column in column_names[6:8]:
#     print(x_column)
#     plt.plot(df[x_column], df[column], label=column)

# plt.plot(X, y)
# plt.legend(loc="best")
# plt.show()

# Split the data into trainning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2023)

# Use Linear Regression to train the model and test the model
model = LinearRegression()

# Use different model
# from sklearn.linear_model import Lasso
# model = Lasso()
# from sklearn.linear_model import Ridge
# model = Ridge()

print("Params:", LinearRegression().get_params())
quit()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse:", rmse)