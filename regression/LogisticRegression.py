import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('regression_exercise.csv')

X = df.drop(['Y'], axis=1)
y = df['Y']

selected_features = X.columns

model = LogisticRegression()
model.fit(X, y)