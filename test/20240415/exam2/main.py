import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

BloodPressure = {'低': 0, '正常': 1, '高': 2}
BloodOxygen = {'低': 0, '正常': 1, '高': 2}
Disease = {'A': 0, 'B': 1, 'C': 2}
disease_name = list(Disease.keys())

data = pd.read_csv('heart.csv')
data = data.dropna()
data = data[~data.apply(lambda row: row.astype(str).str.contains(' |NA|NAN|N|T|F|Y').any(), axis=1)]

X = data.drop(['疾病'], axis=1)
X['血壓'] = X['血壓'].map(BloodPressure)
X['血氧'] = X['血氧'].map(BloodOxygen)

y = data['疾病'].map(Disease)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')
print(cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


DiseaseReverse = {v: k for k, v in Disease.items()}

test_data = pd.read_csv('heart_test.csv')
test_X = test_data.copy()

test_X['血壓'] = test_X['血壓'].map(BloodPressure)
test_X['血氧'] = test_X['血氧'].map(BloodOxygen)

pred_y = model.predict(test_X)

test_data['疾病'] = pred_y
test_data['疾病'] = test_data['疾病'].map(DiseaseReverse)
test_data.to_csv('heart_test_result.csv', index=False)
