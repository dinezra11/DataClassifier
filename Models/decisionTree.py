"""
    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""

from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
from Models.baseModel import BaseModel


class DecisionTree(BaseModel):
    def __init__(self):
        self.target = None
        self.model = None
        self.encoder = None

    def train(self, data, target: str):
        """ Initialize the model's object.

        @:param data         The dataframe for training.
        @:param target       The target attribute to classify.
        """

        self.target = target
        x = data.drop(columns=[self.target])
        y = data[self.target]
        self.encoder = make_column_transformer((OneHotEncoder(), x.columns), remainder="passthrough")

        x = self.encoder.fit_transform(x).toarray()

        self.model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(x, y)

    def test(self, data):
        """ Test the model with a test dataset.

        @:param data       The dataframe to test the model.
        """
        if self.target is None:
            print("Error! Please train the model before trying to testing it.")
            return

        x = data.drop(columns=[self.target])
        y = data[self.target]

        x = self.encoder.transform(x).toarray()

        # Return test measurements
        return self.calculatePerformance(self.model.predict(x), y)

    def predict(self, data: dict):
        self.model.predict(data)
