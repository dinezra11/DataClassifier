""" Base class for a model. All of the project's models' classes will inherit from it.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
from sklearn.metrics import confusion_matrix, silhouette_score

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
        confMat = confusion_matrix(yTrue, yPredict)
        tp, fp, fn, tn = int(confMat[1][1]), int(confMat[0][1]), int(confMat[1][0]), int(confMat[0][0])
        precision, recall = tp / (tp + fp), tp / (tp + fn)

        return {
            "accuracy": (tp + tn) / int(yTrue.size),
            "precision": precision,
            "recall": recall,
            "f_measure": (2*precision*recall) / (precision + recall),
            "confusion matrix": confMat,
            "majorityRule": int(yTrue.value_counts().max()) / int(yTrue.size)
        }

    @staticmethod
    def calculateSilhuoette(data, labels):
        """ Calculate the clustering model's score according to the silhuoette method.

        @:param data            The data for measuring the score.
        @:param model           The labels of the fitted clustering model (K-Means).
        """
        return silhouette_score(data, labels, metric="euclidean")
