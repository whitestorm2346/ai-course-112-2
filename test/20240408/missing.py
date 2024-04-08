import pandas as pd

data = pd.read_csv('iris_miss.csv')

missing_values = data.isnull()
missing_per_feature = missing_values.sum()
missing_count = missing_per_feature.sum()
column_title = list(missing_values.columns)

print(f'總共缺失 {missing_count} 筆資料')

for i in range(len(missing_per_feature)):
    print(f'{column_title[i]} 缺失 {missing_per_feature[i]} 筆資料')

for column in missing_values.columns:
    for index, value in missing_values[missing_values[column] == True].iterrows():
        print(f'第 {index + 2} 個 row 的第 {column_title.index(column) + 1} 個 column')
