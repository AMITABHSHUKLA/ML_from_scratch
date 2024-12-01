import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

import requests
from pathlib import Path
ml_algo = Path("ML_Algorithms")
if not ml_algo.is_dir():
  print("Creating ML_Algorithms directory")
  ml_algo.mkdir()

file_path = ml_algo / "KNN.py"
with open(file_path,"wb") as file:
  request = requests.get("https://raw.githubusercontent.com/AMITABHSHUKLA/ML_from_scratch/refs/heads/main/Machine_learning_from_scratch/KNN.py")
  print("file downloaded")
  print(request)
  file.write(request.content)

from KNN import KNN

from KNN import accuracy

classifier = KNN(5)

classifier.fit(X_train,y_train)

classifier.predict_x(X_train[2])

y_pred = classifier.predict(X_test)

print(y_pred)

print(f"predictions: {len(y_pred)}, Y_test:{len(y_test)},x_test :{len(X_test)}")

accuracy(y_pred,y_test)

