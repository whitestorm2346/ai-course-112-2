from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

weights = ['uniform', 'distance']
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
metrics = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']

for weight in weights:
    for algorithm in algorithms:
        for metric in metrics:
            model_pca = KNeighborsClassifier(n_neighbors=5, weights=weight, algorithm=algorithm, metric=metric)
            model_pca.fit(X_train, y_train)

            y_pred = model_pca.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'[weight: {weight}][algorithm: {algorithm}][metric: {metric}]<accuracy> ', accuracy, end='\n\n')