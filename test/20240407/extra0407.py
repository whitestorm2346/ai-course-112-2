import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class ModelInfo:
    def __init__(self, type, accuracy) -> None:
        self.type = type
        self.accuracy = accuracy

class TestAccuracy:
    def __init__(self, X_train, y_train, X_test, y_test, target_accuracy=-1.0) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.target_accuracy = target_accuracy
        self.model = None

    def get_linear_reg(self) -> list[ModelInfo]:
        fit_intercepts = [True, False]
        results = []

        for element in fit_intercepts:
            self.model = LinearRegression(fit_intercept=element)
            self.model.fit(self.X_train, self.y_train)

            self.y_pred = self.model.predict(self.X_test)
            self.accuracy = r2_score(self.y_pred, self.y_test)

            print('<accuracy>', self.accuracy, end='\r')

            if self.accuracy >= self.target_accuracy:
                results.append(ModelInfo(f'[LinearRegression][fit_intercept: {element}]', self.accuracy))

        return results
    
    def get_logistic_reg(self) -> list[ModelInfo]:
        fit_intercepts = [True, False]
        results = []

        for element in fit_intercepts:
            self.model = LogisticRegression(fit_intercept=element)
            self.model.fit(self.X_train, self.y_train)

            self.y_pred = self.model.predict(self.X_test)
            self.accuracy = accuracy_score(self.y_pred, self.y_test)

            print('<accuracy>', self.accuracy, end='\r')

            if self.accuracy >= self.target_accuracy:
                results.append(ModelInfo(f'[LogisticRegression][fit_intercept: {element}]', self.accuracy))

        return results

    def get_svm(self, max_degree=10) -> list[ModelInfo]:
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = []

        for kernel in kernels:
            if kernel == 'poly':
                for degree in range(2, max_degree + 1):
                    self.model = SVC(kernel=kernel, degree=degree)
                    self.model.fit(self.X_train, self.y_train)

                    self.y_pred = self.model.predict(self.X_test)
                    self.accuracy = accuracy_score(self.y_pred, self.y_test)

                    print('<accuracy>', self.accuracy, end='\r')

                    if self.accuracy >= self.target_accuracy:
                        results.append(ModelInfo(f'[SVM][kernel: {kernel}][degree: {degree}]', self.accuracy))
            elif kernel in ['rbf', 'sigmoid']:
                for gamma in range(1, 101):
                    self.model = SVC(kernel=kernel, gamma=gamma / 100)
                    self.model.fit(self.X_train, self.y_train)

                    self.y_pred = self.model.predict(self.X_test)
                    self.accuracy = accuracy_score(self.y_pred, self.y_test)

                    print('<accuracy>', self.accuracy, end='\r')

                    if self.accuracy >= self.target_accuracy:
                        results.append(ModelInfo(f'[SVM][kernel: {kernel}][gamma: {gamma / 100}]', self.accuracy))
            else:
                self.model = SVC(kernel=kernel)
                self.model.fit(self.X_train, self.y_train)

                self.y_pred = self.model.predict(self.X_test)
                self.accuracy = accuracy_score(self.y_pred, self.y_test)

                print('<accuracy>', self.accuracy, end='\r')

                if self.accuracy >= self.target_accuracy:
                    results.append(ModelInfo(f'[SVM][kernel: {kernel}]', self.accuracy))

        return results

    def get_knn(self, max_n=5) -> list[ModelInfo]:
        weights = ['uniform', 'distance']
        algorithms = ['ball_tree', 'kd_tree', 'brute']
        metrics = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
        results = []

        for n in range(3, max_n + 1):
            for weight in weights:
                for algorithm in algorithms:
                    for metric in metrics:
                        self.model = KNeighborsClassifier(n_neighbors=n, weights=weight, algorithm=algorithm, metric=metric)
                        self.model.fit(self.X_train, self.y_train)

                        self.y_pred = self.model.predict(self.X_test)
                        self.accuracy = accuracy_score(self.y_test, self.y_pred)

                        print('<accuracy> ', self.accuracy, end='\r')

                        if self.accuracy >= self.target_accuracy:
                            results.append(ModelInfo(f'[KNN: {n}][weight: {weight}][algorithm: {algorithm}][metric: {metric}]', self.accuracy))

        return results

    def get_dt(self, max_depth_range=10, min_samples_leaf_range=10) -> list[ModelInfo]:
        criterions = ['gini', 'entropy']
        results = []

        for criterion in criterions:
            for max_depth in range(1, max_depth_range + 1):
                for min_samples_leaf in range(1, min_samples_leaf_range + 1):
                    self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                    self.model.fit(self.X_train, self.y_train)

                    self.y_pred = self.model.predict(self.X_test)
                    self.accuracy = accuracy_score(self.y_pred, self.y_test)

                    print('<accuracy>', self.accuracy, end='\r')

                    if self.accuracy >= self.target_accuracy:
                        results.append(
                            ModelInfo(f'[DecisionTree][criterion: {criterion}][max_depth: {max_depth}][min_samples_leaf: {min_samples_leaf}]'
                                      , self.accuracy)
                    )

        return results

    def get_rf(self, max_estimator=100) -> list[ModelInfo]:
        results = []

        for estimator in range(2, max_estimator + 1):
            self.model = RandomForestClassifier(n_estimators=estimator)
            self.model.fit(self.X_train, self.y_train)

            self.y_pred = self.model.predict(self.X_test)
            self.accuracy = accuracy_score(self.y_pred, self.y_test)

            print('<accuracy>', self.accuracy, end='\r')

            if self.accuracy > self.target_accuracy:
                results.append(ModelInfo(f'[RandomForest][n_estimators: {estimator}]', self.accuracy))

        return results


#===================================================
#                   Main Section    
#===================================================

GENDER = {'女': 0, '男': 1}

train_data_set = pd.read_csv('DT_dataset_1.csv')
test_data_set = pd.read_csv('test_data.csv')

X = train_data_set.drop('療程是否成效', axis=1)
X['性別'] = X['性別'].map(GENDER)
y = train_data_set['療程是否成效']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.2)

test_X = test_data_set.drop('療程是否成效', axis=1)
test_X['性別'] = test_X['性別'].map(GENDER)
test_y = test_data_set['療程是否成效']

get_accuracy = TestAccuracy(X_train, y_train, X_test, y_test, 0.94)
target_models = []

# target_models += get_accuracy.get_linear_reg()
# target_models += get_accuracy.get_logistic_reg()
# target_models += get_accuracy.get_svm(max_degree=10) # very slow
# target_models += get_accuracy.get_knn(max_n=5)
target_models += get_accuracy.get_dt(max_depth_range=25, min_samples_leaf_range=12)
target_models += get_accuracy.get_rf(max_estimator=100)


for model in target_models:
    print(model.type, model.accuracy)

    
