""" This class represents a KNN model, using the implementations from sklearn library.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd

from Models.baseModel import BaseModel


class KNeighbors(BaseModel):
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
        x = data.drop(columns=[self.target])
        y = data[self.target]
        self.encoder = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), x.columns), remainder="passthrough")

        x = self.encoder.fit_transform(x)
        if type(x) != np.ndarray:
            x = x.toarray()

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

        x = self.encoder.transform(x)
        if type(x) != np.ndarray:
            x = x.toarray()

        # Return test measurements
        return self.calculatePerformance(self.model.predict(x), y)

    def predict(self, entry):
        return self.model.predict(entry)
