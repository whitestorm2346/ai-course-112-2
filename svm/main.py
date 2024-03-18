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

# 使用PCA 降維成3 維
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=100, test_size=0.3)

model_pca = SVC(kernel='linear')
model_pca.fit(X_train, y_train)

y_pred = model_pca.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('<accuracy> ', accuracy, end='\n\n')

cm = confusion_matrix(y_test, y_pred)
print("cm",cm)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

print("TP : ",TP)
print("FP : ",FP)
print("FN : ",FN)

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

# 繪製決策邊界
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
z_min, z_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1

# 指定生成的點的數量
num_points = 50

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, num_points),
    np.linspace(y_min, y_max, num_points),
    np.linspace(z_min, z_max, num_points)
)
grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
Z = model_pca.predict(grid)
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, edgecolors='k', cmap=plt.cm.Paired)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D PCA with Labels')

plt.show()
