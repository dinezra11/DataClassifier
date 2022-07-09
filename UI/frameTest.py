""" Model testing page.
    In this page, the user will be able to load a model and test it with a test dataset.
    After the testing, a test measurements will be shown on a new window. (Accuracy, Confusion Matrix, etc.)

    The results will also be saved in the desired directory.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk
from tkinter import filedialog as fd

from Models.decisionTree import DecisionTree
from Models.kMeans import KMeansModel
from Models.kNeighbors import KNeighbors
from Models.modelManager import loadModel
from Models.naiveBayes import NaiveBayes
from Models.ourDecisionTree import OurTree
from Models.ourNaiveBayes import OurNaiveBayes
from UI import windowResults
from preprocessing import loadCsv


class FrameTestUI:
    """ A wrapper class for the testing page.
    The files path will be stored here.
    """

    def __init__(self):
        # Settings Variables
        self.modelPath = None
        self.dataPath = None
        self.folderPath = None

    def inputComponents(self, frame):
        """ Initialize and pack all of the user input components. """

        def modelSelect():
            self.modelPath = fd.askopenfilename()
            try:
                t = type(loadModel(self.modelPath))
                if t == OurNaiveBayes:
                    lblModelResult.config(text="Model Selected: Naive Bayes (Our Implementation)", fg="green")
                elif t == NaiveBayes:
                    lblModelResult.config(text="Model Selected: Naive Bayes (Sklearn's)", fg="green")
                elif t == OurTree:
                    lblModelResult.config(text="Model Selected: ID3 Decision Tree (Our Implementation)", fg="green")
                elif t == DecisionTree:
                    lblModelResult.config(text="Model Selected: ID3 Decision Tree (Sklearn's)", fg="green")
                elif t == KNeighbors:
                    lblModelResult.config(text="Model Selected: K Nearest Neighbors", fg="green")
                elif t == KMeansModel:
                    lblModelResult.config(text="Model Selected: K-Means Clustering", fg="green")
                else:
                    raise Exception()
            except Exception:
                lblModelResult.config(text="Invalid model file..", fg="red")
                self.modelPath = None

        def dataSelect():
            self.dataPath = fd.askopenfilename(filetypes=(('CSV file', '*.csv'),))
            if self.dataPath == "":
                lblDataResult.config(text="No data loaded..", fg="red")
                self.dataPath = None
            else:
                lblDataResult.config(text="Data Loaded!", fg="green")

        def folderSelect():
            self.folderPath = fd.askdirectory()
            if self.folderPath == "":
                lblFolderResult.config(text="No folder selected..", fg="red")
                self.folderPath = None
            else:
                lblFolderResult.config(text="Folder Found!", fg="green")

        btnModel = tk.Button(frame, text="Load Model", command=modelSelect, relief="groove")
        lblModelResult = tk.Label(frame, text="")

        btnData = tk.Button(frame, text="Load Testing Data", command=dataSelect, relief="groove")
        lblDataResult = tk.Label(frame, text="")

        btnFolder = tk.Button(frame, text="Select Folder For Results", command=folderSelect, relief="groove")
        lblFolderResult = tk.Label(frame, text="")

        # Placing widgets
        btnModel.pack()
        lblModelResult.pack()
        btnData.pack(pady=(10, 0))
        lblDataResult.pack()
        btnFolder.pack(pady=(10, 0))
        lblFolderResult.pack()

    def performTesting(self, lblOutput):
        try:
            model = loadModel(self.modelPath)
            data = loadCsv(self.dataPath)
            txtFile = self.folderPath + "/Test Results.txt"

            results = model.test(data)

            report = "Accuracy: " + str(results["accuracy"]) + "\n" + "Precision: " + str(
                results["precision"]) + "\n" + "Recall: " + str(results["recall"]) + "\n" "F Measure: " + str(
                results["f_measure"]) + "\n\n"
            report += "True Positive: " + str(results["confusion matrix"][0][0]) + "\nFalse Positive: " + str(
                results["confusion matrix"][0][1]) + "\nFalse Negative: " + str(
                results["confusion matrix"][1][0]) + "\nTrue Negative: " + str(results["confusion matrix"][1][1])
            with open(txtFile, 'w') as f:
                f.write(str(report))

            windowResults.showResults(results)
        except ValueError as ve:
            msg = "Error! " + str(ve)
            lblOutput.config(text=msg, fg="red")
        except Exception as e:
            print(e)
            lblOutput.config(text="Error! Couldn't perform the testing.", fg="red")
        else:
            lblOutput.config(text="The model has been successfully tested! The results saved in the folder. :)",
                             fg="green")


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def backButton():
        navigateFunction("start")

    def testButton():
        ui.performTesting(lblOutput)

    frame = tk.Frame(window)
    ui = FrameTestUI()
    tk.Label(frame, text="Model Testing", font=(None, 24)).pack()

    # Initialize the widgets
    ui.inputComponents(frame)

    tk.Button(frame, text="Perform Testing!", relief="groove", font=(None, 20), command=testButton).pack(pady=(20, 0))
    lblOutput = tk.Label(frame, text="")
    lblOutput.pack()
    tk.Button(frame, text="Back to Main Menu", relief="groove", font=(None, 12), command=backButton).pack(side="bottom", pady=10)

    return frame
