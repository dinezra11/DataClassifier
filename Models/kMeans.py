""" This class represents a K-Means clustering model, using the implementations from sklearn library.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd

from Models.baseModel import BaseModel


class KMeansModel(BaseModel):
    def __init__(self):
        """ Initialize the model's object. """
        self.model = None

    def train(self, data, n: int):
        """ Initialize the model's object.

        @:param data            The dataframe for training.
        @:param n               Amount of clusters
        """

        self.model = KMeans(n)

    def test(self, data):
        """ Test the model with a test dataset.

        @:param data       The dataframe to test the model.
        """

        encoder = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), data.columns), remainder="passthrough")
        data = encoder.fit_transform(data).toarray()

        labels = self.model.fit(data).labels_
        data = pd.DataFrame(data)
        data["cluster"] = labels

        return data, self.calculateSilhuoette(data, labels)

    def predict(self, entry):
        return self.model.predict(entry)
