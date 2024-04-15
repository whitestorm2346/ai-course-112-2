import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

train = pd.read_csv('house_price.csv')

X = train.drop(['Price'], axis=1)
y = train['Price']

test = pd.read_csv('house_price_test.csv')
result_csv = test

while True:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train)
    results = model.fit()

    temp = temp_i = 0
    print("====p-values====")
    print(results.pvalues)
    print("================\n\n")

    for i in range(1,len(results.pvalues)):
        if temp < results.pvalues[i]:
            temp = results.pvalues[i]
            temp_i = i - 1
    if temp > 0.05:
        print(f"第{temp_i + 1}被刪掉 p值 : {temp}")
        X = X.drop(columns=[X.columns[temp_i]])

        print("=====data frames=====")
        print(X)
        print("=====================\n\n")

        temp = temp_i = 0
    else:
        break


# print(results.summary())

coef_column = results.params
columns = list(X.columns)

print('Price = ', coef_column[0], end=' ')

for i in range(len(columns)):
    print(f'+ {coef_column[i + 1]}*{columns[i]}', end=' ')

print()

test = test[columns]
test = sm.add_constant(test)

print(test)

print(X_train)

y_pred = results.predict(test)

result_csv['Predicted'] = y_pred
result_csv.to_csv('house_price_result.csv', index=False)
    