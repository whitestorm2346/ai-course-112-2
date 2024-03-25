import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

GENDER = {'女': 0, '男': 1}

train_data = pd.read_csv('DT_dataset_1.csv')
test_data = pd.read_csv('test_data.csv')

X_train = train_data.drop('療程是否成效', axis=1)
X_train['性別'] = X_train['性別'].map(GENDER)
y_train = train_data['療程是否成效']

X_test = test_data.drop('療程是否成效', axis=1)
X_test['性別'] = X_test['性別'].map(GENDER)
y_test = test_data['療程是否成效']

criterions = ['gini', 'entropy']
max_depth = 1
min_samples_leaf = 1

while True:
    for criterion in criterions:
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'[{criterion}]<accuracy> ', accuracy, end='\n\n')

    print('(1) [max_depth] + 1')
    print('(2) [max_depth] - 1')
    print('(3) [min_samples_leaf] + 1')
    print('(4) [min_samples_leaf] - 1')
    print('(5) end program')

    code = int(input('Option: '))
