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
import tkinter as tk


def showResults(results: dict, txtFile: str):
    """ Create a new window for showing the results of the model testing.

    @:param results         The dictionary of the testint results.
    @:param txtFile         The path to the file where to save the results.
    """

    def frameMatrix(root):
        """ Draw the Confusion Matrix. """
        mat = results["confusion matrix"]
        tp, fp, fn, tn = str(mat[0][0]), str(mat[0][1]), str(mat[1][0]), str(mat[1][1])
        frame = tk.Frame(root)

        tk.Label(frame, text="Actual", fg="gray").grid(column=2, row=0, columnspan=2)
        tk.Label(frame, text="True").grid(column=2, row=1)
        tk.Label(frame, text="False").grid(column=3, row=1)

        tk.Label(frame, text=tp, fg="green").grid(column=2, row=2)
        tk.Label(frame, text=fp, fg="red").grid(column=3, row=2)
        tk.Label(frame, text=fn, fg="red").grid(column=2, row=3)
        tk.Label(frame, text=tn, fg="green").grid(column=3, row=3)

        tk.Label(frame, text="Prediction", fg="gray").grid(column=0, row=2)
        tk.Label(frame, text="True").grid(column=1, row=2)
        tk.Label(frame, text="False").grid(column=1, row=3)

        return frame

    # Save results
    report = "Accuracy: " + str(results["accuracy"]) + "\n" + "Precision: " + str(
        results["precision"]) + "\n" + "Recall: " + str(results["recall"]) + "\n" "F Measure: " + str(
        results["f_measure"]) + "\n\n"
    confMat = "True Positive: " + str(results["confusion matrix"][0][0]) + "\nFalse Positive: " + str(
        results["confusion matrix"][0][1]) + "\nFalse Negative: " + str(
        results["confusion matrix"][1][0]) + "\nTrue Negative: " + str(results["confusion matrix"][1][1])
    with open(txtFile, 'w') as f:
        f.write(str(report) + str(confMat))

    # Show results in new tkinter's window
    window = tk.Tk()
    window.title("Model's Score")
    window.resizable(False, False)
    tk.Label(window, text=report).pack()
    frameMatrix(window).pack(pady=10)


def showClustering(train, test):
    """ Plot a scatterplot for the k-means clustering. """
    '''pca = PCA(n_components=2)
    features = pd.DataFrame(pca.fit_transform(data.drop(columns=["cluster"])))

    sns.scatterplot(x=features[0], y=features[1], hue=data["cluster"])
    plt.show()'''

    pca = PCA(n_components=2)
    featuresTrain = pd.DataFrame(pca.fit_transform(train.drop(columns=["cluster"])))
    featuresTest = pd.DataFrame(pca.fit_transform(test.drop(columns=["cluster"])))

    f, axes = plt.subplots(1, 2)
    sns.scatterplot(x=featuresTrain[0], y=featuresTrain[1], hue=train["cluster"], ax=axes[0])
    sns.scatterplot(x=featuresTest[0], y=featuresTest[1], hue=test["cluster"], ax=axes[1])
    plt.show()
