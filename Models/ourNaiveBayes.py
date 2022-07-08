""" Developing a class for implementing the Naive Bayes algorithm.
    The class can be used for loading the dataset and training the new model,
    saving the trained model and loading the existing model for predicting new
    data.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
from math import sqrt, pi
import pandas as pd
import numpy as np
from Models.baseModel import BaseModel


class OurNaiveBayes(BaseModel):
    def __init__(self):
        """ Initialize the model's object."""
        # Create an empty dictionary for the attributes probabilities.
        self.probs = dict()
        self.target = None

    def train(self, df, target: str):
        """ Train the new model according to the desired class, and save the trained
        model for later prediction use.

        @:param df         The dataframe for training.
        @:param target     The class attribute for the classification.
        """
        # Check if the classAtr exists in the dataframe's columns
        if target not in df.columns:
            print("Error! Invalid input for the class attribute.\n")
            return

        self.target = target  # The desired class attribute
        cols = df.drop(columns=[target]).columns  # Get all columns EXCEPT the class attribute
        targetValues = df[target].unique()  # Get all unique values of the target class

        # Calculate probabilities for the class attribute
        self.probs[target] = dict()
        targetDf = df[target].value_counts()
        for u in targetValues:
            self.probs[target][u] = targetDf[u] / targetDf.sum()

        # Calculate probabilities for every column
        for c in cols:
            self.probs[c] = dict()
            if df[c].dtype not in [np.int64, np.float64]:
                # Categorical Column
                for u in df[c].unique():
                    self.probs[c][u] = dict()
                    for t in targetValues:
                        self.probs[c][u][t] = (df[df[c] == u][df[target] == t].shape[0] + 1) / (
                                targetDf[t] + len(df[c].unique()))  # Add 1 for Laplacian fix

                    # Add for the Laplacian fix
                    self.probs[c]['__other__'] = dict()
                for t in targetValues:
                    self.probs[c]['__other__'][t] = 1 / (targetDf[t] + len(df[c].unique()))
            else:
                # Numeric Column
                for t in targetValues:
                    tempDf = df[df[target] == t]
                    self.probs[c][t] = (tempDf[c].mean(), tempDf[c].var())

    def test(self, df):
        """ Test the model with a test dataset.

        @:param df       The dataframe to test the model.
        """
        if self.target is None:
            print("Error! Please train the model before trying to testing it.")
            return

        '''success = 0
        failed = 0

        for _, row in df.iterrows():
            answer = row[self.target]
            prediction = self.predict(row.drop(labels=[self.target]))

            if answer == prediction:
                success += 1
            else:
                failed += 1

        print("Success:\t", success)
        print("Failed:\t\t", failed)
        print("Accuracy Rate: ", (success / (success + failed)) * 100, "%")'''

        prediction = []
        for _, row in df.iterrows():
            prediction.append([self.predict(row.drop(labels=[self.target]))])

        # Return test measurements
        return self.calculatePerformance(prediction, df[self.target])

    def predict(self, data: dict):
        """ Make a prediction for a specific data entries.

        @:param data       A dictionary of data, which the model need to predict the
                          result for.

        @:return           The prediction for the class attribute.
        """

        def gaussianDist(x, mean, variance):
            """ Calculate the probability according to the gaussian distribution.

            @:param x          The value of the tested entry.
            @:param mean       Mean. (Average)
            @:param variance   Variance. (standart deviation^2)

            @:return           The probability.
            """
            from math import e

            result = 1 / sqrt(2 * pi * variance)
            result *= e ** ((-((x - mean)) ** 2) / (2 * variance))
            return result

        if self.target is None:
            print("Error! Train the model before trying to predict a value.")
            return

        try:
            maxProb = 0
            maxClass = None
            targetClasses = list(self.probs[self.target].keys())

            for c in targetClasses:
                # Base probability
                p = self.probs[self.target][c]

                # Multiply by the attributes' probabilities
                for k, v in data.items():
                    if (self.probs[k].get(c) is not None) and (type(self.probs[k][c]) is not dict):
                        # The data for this entry is a mean and variance.
                        p *= gaussianDist(v, self.probs[k][c][0], self.probs[k][c][1])
                    else:
                        if self.probs[k].get(v) is not None:
                            p *= self.probs[k][v][c]
                        else:
                            # Fix zero prediction using the Laplacian method
                            p *= self.probs[k]['__other__'][c]

                # Compare results
                if p > maxProb:
                    maxProb = p
                    maxClass = c

            # Return result
            return maxClass
        except Exception as e:
            print("Error occured. Data input is invalid.\n")
