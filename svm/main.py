import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv('Fish.csv')

SPECIES = {
    'Bream': 0, 
    'Roach': 1, 
    'Whitefish': 2, 
    'Parkki': 3, 
    'Perch': 4, 
    'Pike': 5, 
    'Smelt': 6
}
species_name = list(SPECIES.keys())

X = data.drop(['Species'], axis=1)
y = data['Species'].map(SPECIES)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('<accuracy> ', accuracy, end='\n\n')

cm = confusion_matrix(y_test, y_pred)

TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

precision = np.zeros(len(cm))
recall = np.zeros(len(cm))
specificity = np.zeros(len(cm))

for i in range(len(cm)):
    if TP[i] + FP[i] > 0:
        precision[i] = TP[i] / (TP[i] + FP[i])
    else:
        precision[i] = 0
    
    if TP[i] + FN[i] > 0:
        recall[i] = TP[i] / (TP[i] + FN[i])
    else:
        recall[i] = 0
    
    TN = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))

    if TN + FP[i] > 0:
        specificity[i] = TN / (TN + FP[i])
    else:
        specificity[i] = 0


for i in range(len(cm)):
    print(f"[{species_name[i]}]")
    print(f"  Precision: {precision[i]}")
    print(f"  Recall: {recall[i]}")
    print(f"  Specificity: {specificity[i]}", end='\n\n')


sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
