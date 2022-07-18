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
import numpy as np
import tkinter as tk

import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def frameResults(root, results: dict, txtFile: str):
    """ Create a new window for showing the results of the model testing.

    @:param results         The dictionary of the testint results.
    @:param txtFile         The path to the file where to save the results.
    """

    def frameMatrix():
        """ Draw the Confusion Matrix. """
        mat = results["confusion matrix"]
        tp, fp, fn, tn = str(mat[1][1]), str(mat[0][1]), str(mat[1][0]), str(mat[0][0])
        frame = tk.Frame(subFrame)

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
        results["f_measure"]) + "\n" + "Majority Rule Accuracy: " + str(results["majorityRule"]) + "\n"
    confMat = "True Positive: " + str(results["confusion matrix"][1][1]) + "\nFalse Positive: " + str(
        results["confusion matrix"][0][1]) + "\nFalse Negative: " + str(
        results["confusion matrix"][1][0]) + "\nTrue Negative: " + str(results["confusion matrix"][0][0])
    with open(txtFile, 'w') as f:
        f.write(str(report) + str(confMat))

    # Show results in new tkinter's window
    subFrame = tk.Frame(root)
    tk.Label(subFrame, text=report).pack()
    frameMatrix().pack()

    return subFrame


def savePreProcess(filePath: str, config: dict):
    """ Save and export the pre-processing stage configurations of the model as a text file.

    @:param filePath         The path to the file.
    @:param config           The dictionary with all of the input configuration for the pre-processing.
    """
    # Dispatch information from the dictionary
    info = ["Model Type\t->\t" + str(config["model"])]
    if config["missing"]:
        info.append("Missing values\t->\tFill according to the target class.")
    else:
        info.append("Missing values\t->\tFill according to the whole dataset.")

    info.append("Normalization\t->\t" + str(config["norm"]))
    info.append("Discretization\t->\t" + str(config["disc"]))
    info.append("Train-set size\t->\t" + str(config["splitRatio"]))
    if config.get("k") is not None:
        info.append("K value\t\t->\t" + str(config["k"]))

    # Create the string
    out = "\t\t##Pre-Processing Configurations##\n"
    for i in info:
        out += i + "\n"

    # Write to the file
    with open(filePath, 'w') as f:
        f.write(out)


def showResults(train: dict, test: dict, config: dict, filePath: str):
    """ Show results in new tkinter's window.

    @:param train           The train set.
    @:param test            The test set.
    @:param config          The pre-processing stage configurations.
    @:param filePath        Path of where to save and exports the files.
    """
    window = tk.Tk()
    window.title("Model's Score")
    window.resizable(False, False)
    tk.Label(window, text="Model's score on Train-Set", font=(None, 18)).pack()
    frameResults(window, train, filePath + "/Train Results.txt").pack(pady=(5, 15))
    tk.Label(window, text="Model's score on Test-Set", font=(None, 18)).pack(pady=(5, 0))
    frameResults(window, test, filePath + "/Test Results.txt").pack(pady=5)

    savePreProcess(filePath + "/Pre Processing Config.txt", config)

    window.mainloop()


def showClustering(train, test, scoreTrain: np.float, scoreTest: np.float, config: dict, filePath: str):
    """ Plot a scatterplot for the k-means clustering. """
    titles = ["Train Silhuoette Score: " + str(scoreTrain.__round__(4)),
              "Test Silhuoette Score: " + str(scoreTest.__round__(4))]
    pca = PCA(n_components=2)
    featuresTrain = pd.DataFrame(pca.fit_transform(train.drop(columns=["cluster"])))
    featuresTest = pd.DataFrame(pca.fit_transform(test.drop(columns=["cluster"])))

    f, axes = plt.subplots(1, 2)
    sns.scatterplot(x=featuresTrain[0], y=featuresTrain[1], hue=train["cluster"], ax=axes[0]).set(title=titles[0])
    sns.scatterplot(x=featuresTest[0], y=featuresTest[1], hue=test["cluster"], ax=axes[1]).set(title=titles[1])
    plt.show()

    savePreProcess(filePath + "/Pre Processing Config.txt", config)
    with open(filePath + "/Clustering Results.txt", 'w') as f:
        f.write("Silhouette score for train-set: " + str(scoreTrain.__round__(4)) +
                "\nSilhouette score for test-set: " + str(scoreTest.__round__(4)))
