from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 載入iris 資料集
df = pd.read_csv('Fish.csv')
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Species'].map({'Bream':0,'Roach':1,'Whitefish':2,'Parkki':3,'Perch':4,'Pike':5,'Smelt':6})




# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 使用PCA 降維成2 維
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

# 建立SVM 模型
model = SVC() # kernel='rbf'

# 訓練模型
model.fit(X_train_pca, y_train)

# 預測測試集
y_pred = model.predict(X_test_pca)

# 計算精準度
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# 繪製預測分布圖
# h = .02  # step size in the mesh
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# 繪製訓練點
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision boundary')
plt.show()

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
# 繪製混淆矩陣圖
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()