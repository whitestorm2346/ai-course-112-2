from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ModelInfo:
    def __init__(self, type, accuracy) -> None:
        self.type = type
        self.accuracy = accuracy

class TestAccuracy:
    def __init__(self, X_train, y_train, X_test, y_test, target_accuracy=-1) -> None:
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