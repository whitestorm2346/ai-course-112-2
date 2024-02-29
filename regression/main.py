import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict


df = pd.read_csv('regression_exercise.csv')

# X = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
# X = df[['A', 'C', 'D', 'E', 'F', 'G', 'H']]
# X = df[['A', 'C', 'D', 'E', 'F', 'H']]
# X = df[['A', 'C', 'E', 'F', 'H']]
X = df[['A', 'C', 'E', 'F']]
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = model.predict(X_test)

# 添加截距項
X = sm.add_constant(X)

# 初始化線性回歸模型
model = sm.OLS(y, X)

# 用最小二乘法擬合模型
results = model.fit()

# 模型摘要，包括係數和 p 值
print(results.summary())

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

coef_column = results.params

print(f"regression: {coef_column[0]}", end='')

for i in range(1, len(X.columns)):
    print(f' + {coef_column[i]}*{X.columns[i]}', end='')

print()

mean_y_train = y_train.mean()
mean_y_pred = y_pred.mean()

bias = mean_y_train - mean_y_pred

print("Bias :", bias)

# 使用 k 折交叉驗證計算模型的預測值
y_pred_cv = cross_val_predict(LinearRegression(), X_test, y_test, cv=5)  # 假設使用 5 折交叉驗證

variance = np.var(y_pred_cv)

print("Variance :", variance)