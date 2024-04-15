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

# 讀取 CSV 檔案並將其轉換為 DataFrame
df = pd.read_csv('20230304exam.csv')

# 分離特徵和目標變數

X = df[['A' ,'B','C','D' , 'E', 'F','G','H', 'I' ]]                        
y = df['Y']

print("=====data frames=====")
print(X)
print("=====================")

while True:
    # 分割資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    # 添加截距項
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    # 初始化線性回歸模型
    model = sm.OLS(y_train, X_train)
    
    # 用最小二乘法擬合模型
    results = model.fit()
    temp = temp_i = 0
    print("====p-values====")
    print(results.pvalues)
    print("================")

    for i in range(1,len(results.pvalues)):
        if temp < results.pvalues[i]:
            temp = results.pvalues[i]
            temp_i = i - 1
    if temp > 0.05:
        print(f"第{temp_i + 1}被刪掉 p值 : {temp}")
        X = X.drop(columns=[X.columns[temp_i]])
        print("=====data frames=====")
        print(X)
        print("=====================")
        temp = temp_i = 0
    else:
        break
# 模型摘要，包括係數和 p 值

print(results.summary())


# 計算模型的性能指標（例如 MSE、MAE、RMSE）
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

# 提取係數（coef）列
coef_column = results.params

# 顯示係數（coef）列

print(f"regression : {coef_column[0]} + {coef_column[1]}*A + {coef_column[2]}*C + {coef_column[3]}*E + {coef_column[4]}*F")

# 計算訓練集的平均目標值
mean_y_train = y_train.mean()

# 計算模型的預測值的平均值
mean_y_pred = y_pred.mean()

# 計算偏差
bias = mean_y_train - mean_y_pred

print("Bias :", bias)

# 使用 k 折交叉驗證計算模型的預測值
y_pred_cv = cross_val_predict(LinearRegression(), X_test, y_test, cv=5)  # 假設使用 5 折交叉驗證

# 計算模型預測值的方差
variance = np.var(y_pred_cv)

print("Variance :", variance)
