import pandas as pd

data = pd.read_csv('iris_abnormal.csv')
threshold = 4

abnormal_values = data[(data - data.mean()).abs() > threshold * data.std()]
abnormal_per_feature = abnormal_values.count()
abnormal_count = abnormal_per_feature.sum()

column_title = list(abnormal_values.columns)

print(f'總共異常 {abnormal_count} 筆資料')

for i in range(len(abnormal_per_feature)):
    print(f'{column_title[i]} 缺失 {abnormal_per_feature[i]} 筆資料')

for column in abnormal_values.columns:
    for index, value in abnormal_values[column].items():
        if not pd.isnull(value):
            print(f'第 {index + 2} 個 row 的第 {column_title.index(column) + 1} 個 column -> 異常值: {value}')
