from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier()
model.fit(X, y)

print("Model trained successfully")