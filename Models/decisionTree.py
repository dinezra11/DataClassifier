"""
    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""

from sklearn import tree
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(self, path: str = None):
        self.model = None

    def train(self, x, y):
        self.model = tree.DecisionTreeClassifier().fit(x, y)

    def test(self, path: str):
        pass

    def predict(self, data: dict):
        self.model.predict(data)
