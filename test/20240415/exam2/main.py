import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

BloodPressure = {'低': 0, '正常': 1, '高': 2}
BloodOxygen = {'低': 0, '正常': 1, '高': 2}
Disease = {'A': 0, 'B': 1, 'C': 2}

data = pd.read_csv('heart.csv')

X = data.drop(['疾病'], axis=1)
X['血壓'] = X['血壓'].map(BloodPressure)
X['血氧'] = X['血氧'].map(BloodOxygen)

y = data['疾病'].map(Disease)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)