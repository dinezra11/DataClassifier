""" This file creates a new tkinter window for showing the model testing results.
    The function "showResults(dict)" will be called from the frameTest file, after performing the testing on the model.

    The results that are going to be shown are: Accuracy
                                                Precision
                                                Recall
                                                F Measure
                                                Confusion Matrix

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plotGraph(confusionMatrix):
    pass

def showResults(results: dict, txtFile: str):
    """ Create a new window for showing the results of the model testing.

    @:param results         The dictionary of the testint results.
    @:param txtFile         The path to the file where to save the results.
    """
    # Save Results
    report = "Accuracy: " + str(results["accuracy"]) + "\n" + "Precision: " + str(
        results["precision"]) + "\n" + "Recall: " + str(results["recall"]) + "\n" "F Measure: " + str(
        results["f_measure"]) + "\n\n"
    report += "True Positive: " + str(results["confusion matrix"][0][0]) + "\nFalse Positive: " + str(
        results["confusion matrix"][0][1]) + "\nFalse Negative: " + str(
        results["confusion matrix"][1][0]) + "\nTrue Negative: " + str(results["confusion matrix"][1][1])
    with open(txtFile, 'w') as f:
        f.write(str(report))


def showClustering(data):
    """ Plot a scatterplot for the k-means clustering. """
    pca = PCA(n_components=2)
    features = pd.DataFrame(pca.fit_transform(data.drop(columns=["cluster"])))

    sns.scatterplot(x=features[0], y=features[1], hue=data["cluster"])
    plt.show()
