import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

# pd.set_option('display.max_rows', None)

df = pd.read_csv(r'sklearn\imbd-template\2022_first\2022-train-v2.csv')

df = df.dropna(axis=1)
# print(df.isna().sum())

# assume the single output
y0 = df.iloc[:, 0]
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]
y3 = df.iloc[:, 3]
y4 = df.iloc[:, 4]
y5 = df.iloc[:, 5]

X = df.iloc[:, 6:]

# separate  the features to clean, oven, painting, env
results = df.iloc[:, :6]
clean_features = df.loc[:, 'clean_temp':'clean_pressure102']
oven_features = df.loc[:, 'oven_pa1':'oven_b3']
painting_features = df.loc[:, 'painting_g1_act_a_air':'painting_g12_act_hvc']
env_features = df.loc[:, 'env_rpi05_hum':'env_rpi15_temp']
print("before PCA\n", oven_features)

# try to do some PCA
new_oven_features = PCA(n_components=3).fit_transform(oven_features)
# print("after PCA\n", new_oven_features)
columns = ['oven1', 'oven2', 'oven3']
new_oven_features = pd.DataFrame(new_oven_features, columns=columns)
# print(new_oven_features)

new_clean_features = PCA(n_components=5).fit_transform(oven_features)
columns = ['clean1', 'clean2', 'clean3', 'clean4', 'clean5']
new_clean_features = pd.DataFrame(new_clean_features, columns=columns)

new_painting_features = PCA(n_components=8).fit_transform(painting_features)
columns = ['painting1', 'painting2', 'painting3', 'painting4', 'painting5', 'painting6', 'painting7', 'painting8']
new_painting_features = pd.DataFrame(new_painting_features, columns=columns)

new_env_features = PCA(n_components=5).fit_transform(env_features)
columns = ['env1', 'env2', 'env3', 'env4', 'env5']
new_env_features = pd.DataFrame(new_env_features, columns=columns)

data_features = pd.concat([new_clean_features, new_oven_features, new_painting_features, new_env_features], axis=1, ignore_index=True)


pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressor', RandomForestRegressor())
])

parameter = {

}

shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2023)

grid_search = GridSearchCV(pipeline, parameter, cv=shuffle_split,
    scoring='neg_root_mean_squared_error', return_train_score=True, verbose=3)

y_outputs = [y0, y1, y2, y3, y4, y5]
y_score = []

for idx, y_output in enumerate(y_outputs):
    grid_search.fit(data_features, y_output)
    y_score.append(-grid_search.best_score_)

avg_score = np.mean(y_score)

print("y_score: ", y_score)
print("avg rmse score: ", avg_score)