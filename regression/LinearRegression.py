import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict


df = pd.read_csv('regression_exercise.csv')

X = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
# X = df[['A', 'B', 'E', 'F', 'G', 'H']]
# X = df[['A', 'B', 'F', 'G', 'H']]
# X = df[['A', 'B', 'F', 'G']]
# X = df[['A', 'B', 'F']]
# X = df[['A', 'B']]
# X = df[['A']]
y = df['Y']

model = LinearRegression()
model.fit(X, y)

y = y.ravel()

F_values, p_values = f_regression(X, y)

for i in range(len(F_values)):
    print(X.columns[i], ':')
    print('f = %f' % F_values[i])
    print('p = %f' % p_values[i])
    print()


print('[intercept]:', model.intercept_)
print('[coef_]:', model.coef_)


# ============= 測試模型 ==================
# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print("MSE:", mse)
# print("MAE:", mae)
# print("RMSE:", rmse, end='\n\n')

# mean_y_train = y_train.mean()
# mean_y_pred = y_pred.mean()

# bias = mean_y_train - mean_y_pred

# print("Bias:", bias)

# y_pred_cv = cross_val_predict(model, X_test, y_test, cv=5) 

# variance = np.var(y_pred_cv)

# print("Variance:", variance)