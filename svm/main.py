import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv('Fish.csv')

X = data.drop(['Y'], axis=1)
y = data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

model = SVC(kernel='rbf', gamma=0.01)
