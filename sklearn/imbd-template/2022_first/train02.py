import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\User\sklearn\sklearn\imbd-template\2022_first\2022-train-v2.csv')

# Erase the columns with missing values
df = df.dropna(axis=1)

# Set y as the target variable, and X is the feature variables
y = df.iloc[:, 0]
X = df.iloc[:, 6:]

# Create a pipeline with scaler and regressor
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', LinearRegression())
])

# Define the parameters for grid search
parameters = {
    'scaler__feature_range': [(0, 1), (0, 10)],
    'regressor__fit_intercept': [True, False],
    'regressor__normalize': [True, False]
}

# Create a ShuffleSplit object for randomizing train and test datasets
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2023)

# Create a GridSearchCV object with the pipeline, parameters, and shuffle split
grid_search = GridSearchCV(pipeline, parameters, cv=shuffle_split,
    scoring='neg_root_mean_squared_error', return_train_score=True, verbose=3)

# Fit the GridSearchCV object to find the best parameters
print("Fit X, y")
grid_search.fit(X, y)

print("Results from GridSearchCV")
# Access the best parameters, best score, and best estimator
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_estimator = grid_search.best_estimator_

# Print the best parameters, best score, and best estimator
print("Best parameters: ", best_params)
print("Best RMSE score: ", best_score)
print("Best estimator: ", best_estimator)
