""" Base class for a model. All of the project's models' classes will inherit from it.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
from Models.ModelManager import saveModel


class BaseModel:
    def save(self, filename: str):
        """ Save the trained model for future use.

        @:param filename       The name of the new file to save.
        """
        saveModel(filename, self)

    # Methods for inherited class implementations:
    def train(self, df, target: str):
        pass

    def test(self, df):
        pass

    def predict(self, data: dict):
        pass
