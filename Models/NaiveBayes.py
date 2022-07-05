""" This class represents a Naive Bayse model, using the implementations from sklearn library.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd


class NaiveBayes:
    def __init__(self):
        """ Initialize the model's object. """
        self.model = None
        self.target = None

    def train(self, data, target: str):
        """ Initialize the model's object.

        @:param data         The dataframe for training.
        @:param target       The target attribute to classify.
       """

        gnb = GaussianNB()
        self.model = gnb.fit(data.drop(columns=[target]), data[target])

    def save(self, filename: str):
        pass

    def test(self, path: str):
        pass

    def predict(self, data: dict):
        self.model.predict(data)