""" This class represents a KNN model, using the implementations from sklearn library.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class KNeighbors:
    def __init__(self):
        """ Initialize the model's object. """
        self.model = None
        self.target = None
        self.encoder = None

    def train(self, data, target: str, k: int):
        """ Initialize the model's object.

        @:param data            The dataframe for training.
        @:param target          The target attribute to classify.
        @:param k               Amount of neighbors (k parameter).
        """

        self.target = target
        self.encoder = LabelEncoder()
        x = data.drop(columns=[self.target])
        y = data[self.target]

        for c in x.columns:
            x[c] = self.encoder.fit_transform(x[c])

        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(x, y)

    def test(self, data):
        """ Test the model with a test dataset.

        @:param data       The dataframe to test the model.
        """
        if self.target is None:
            print("Error! Please train the model before trying to testing it.")
            return

        x = data.drop(columns=[self.target])
        y = data[self.target]

        for c in x.columns:
            x[c] = self.encoder.fit_transform(x[c])

        print(self.model.predict(x))

    def predict(self, entry):
        return self.model.predict(entry)


le = LabelEncoder()
train = pd.read_csv("C:/Users/dinez/PycharmProjects/DataMiningProject/train_clean.csv")
test = pd.read_csv("C:/Users/dinez/PycharmProjects/DataMiningProject/test_clean.csv")
m = KNeighbors()
m.train(train, "class", 3)
m.test(test)
