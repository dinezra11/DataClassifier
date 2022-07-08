""" Base class for a model. All of the project's models' classes will inherit from it.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
from Models.modelManager import saveModel


class BaseModel:
    def save(self, filename: str):
        """ Save the trained model for future use.

        @:param filename       The name of the new file to save.
        """
        saveModel(filename, self)

    @staticmethod
    def calculatePerformance(yPredict, yActual):
        """ Give the model's predictions from the test set and the actual values of the target class, calculate and
        measure the performance of the model and return the details as a dictionary.

        @:param yPredict                The model's predictions.
        @:param yActual                 The actual values of the target class
        """

        def accuracy():
            pass

        def precision():
            pass

        def recall():
            pass

        def f_measure():
            pass

        def confusionMatrix():
            pass

        return {
            "accuracy": accuracy(),
            "precision": precision(),
            "recall": recall(),
            "f_measure": f_measure(),
            "confusion matrix": confusionMatrix()
        }
