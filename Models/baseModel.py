""" Base class for a model. All of the project's models' classes will inherit from it.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, silhouette_score

from Models.modelManager import saveModel


class BaseModel:
    def save(self, filename: str):
        """ Save the trained model for future use.

        @:param filename       The name of the new file to save.
        """
        saveModel(filename, self)

    @staticmethod
    def calculatePerformance(yPredict, yTrue):
        """ Get the model's predictions from the test set and the actual values of the target class, calculate and
        measure the performance of the model and return the details as a dictionary.

        @:param yPredict        The model's predictions.
        @:param yTrue           The actual values of the target class
        """

        return {
            "accuracy": accuracy_score(yTrue, yPredict),
            "precision": precision_score(yTrue, yPredict, average='macro'),
            "recall": recall_score(yTrue, yPredict, average='macro'),
            "f_measure": f1_score(yTrue, yPredict, average='macro'),
            "confusion matrix": confusion_matrix(yTrue, yPredict)
        }

    @staticmethod
    def calculateSilhuoette(data, labels):
        """ Calculate the clustering model's score according to the silhuoette method.

        @:param data            The data for measuring the score.
        @:param model           The labels of the fitted clustering model (K-Means).
        """
        return silhouette_score(data, labels, metric="euclidean")
