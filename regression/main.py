import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = pd.read_csv('regression_exercise.csv')

y = data['Y']
X = data.drop(['Y'], axis=1)

print(X)

model = LinearRegression()
model.fit(X, y)

y = y.ravel()
f_values, p_values = f_regression(X, y)

for i in range(len(f_values)):
    print(X.columns[i], ':')
    print("f-value=%.3f" % (f_values[i]))
    print("p-value=%.3f" % (p_values[i]))
    
    if( p_values[i] < 0.05):
        print(X.columns[i], ' is significant')
    else:
        print(X.columns[i], ' is nonsignificant')

    print()

