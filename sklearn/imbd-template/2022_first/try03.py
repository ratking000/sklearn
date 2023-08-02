import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', None)

df = pd.read_csv(r'sklearn\imbd-template\2022_first\2022-train-v2.csv')

df = df.dropna(axis=1)
print(df.isna().sum())

quit()

# assume the single output
y = df.iloc[:, 0]
X = df.iloc[:, 6:]

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', RandomForestRegressor())
])

parameter = {

}

shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2023)

grid_search = GridSearchCV(pipeline, parameter, cv=shuffle_split,
    scoring='neg_root_mean_squared_error', return_train_score=True, verbose=3)

print("Fit X, y")
grid_search.fit(X, y)

print("Results from GridSearchCV")

best_params = grid_search.best_params_
best_score = -grid_search.best_score_
best_estimator = grid_search.best_estimator_

print("Best RMSE score: ", best_score)
