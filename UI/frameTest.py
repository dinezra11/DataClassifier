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
from UI.colors import *


class FrameTestUI:
    """ A wrapper class for the testing page.
    The files path will be stored here.
    """

    def __init__(self):
        # Settings Variables
        self.modelPath = None
        self.trainPath = None
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

        def trainSelect():
            self.trainPath = fd.askopenfilename(filetypes=(('CSV file', '*.csv'),))
            if self.trainPath == "":
                lblTrainResult.config(text="No data loaded..", fg="red")
                self.trainPath = None
            else:
                lblTrainResult.config(text="Data Loaded!", fg="green")

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

        btnModel = tk.Button(frame, text="Load Model", command=modelSelect, relief="groove", bg=BUTTON_BACK, fg=BUTTON_TEXT)
        lblModelResult = tk.Label(frame, text="", bg=BACKGROUND_COLOR)

        dataFrame = tk.Frame(frame, bg=BACKGROUND_COLOR)
        btnTrain = tk.Button(dataFrame, text="Load Training Data", command=trainSelect, relief="groove", bg=BUTTON_BACK,fg=BUTTON_TEXT)
        lblTrainResult = tk.Label(dataFrame, text="", bg=BACKGROUND_COLOR)
        btnData = tk.Button(dataFrame, text="Load Testing Data", command=dataSelect, relief="groove", bg=BUTTON_BACK, fg=BUTTON_TEXT)
        lblDataResult = tk.Label(dataFrame, text="", bg=BACKGROUND_COLOR)
        btnTrain.grid(column=0, row=0, padx=10)
        lblTrainResult.grid(column=0, row=1)
        btnData.grid(column=1, row=0, padx=10)
        lblDataResult.grid(column=1, row=1)

        btnFolder = tk.Button(frame, text="Select Folder For Results", command=folderSelect, relief="groove", bg=BUTTON_BACK, fg=BUTTON_TEXT)
        lblFolderResult = tk.Label(frame, text="", bg=BACKGROUND_COLOR)

        # Placing widgets
        btnModel.pack()
        lblModelResult.pack()
        dataFrame.pack(pady=(10, 0))
        btnFolder.pack(pady=(10, 0))
        lblFolderResult.pack()

    def performTesting(self, lblOutput):
        try:
            if self.folderPath is None:
                raise ValueError("No folder selected.")
            model = loadModel(self.modelPath)
            train = loadCsv(self.trainPath)
            test = loadCsv(self.dataPath)

            if type(model) == KMeansModel:
                resultsTrain, silhuoetteTrain = model.test(train)
                resultsTest, silhuoetteTest = model.test(test)
            else:
                resultsTrain = model.test(train)
                resultsTest = model.test(test)

            lblOutput.config(text="The model has been successfully tested! The results saved in the folder. :)",
                             fg="green")

            if type(model) == KMeansModel:
                windowResults.showClustering(resultsTrain, resultsTest, silhuoetteTrain, silhuoetteTest, model.preprocess, self.folderPath)
            else:
                windowResults.showResults(resultsTrain, resultsTest, model.preprocess, self.folderPath)
        except ValueError as ve:
            msg = "Error! " + str(ve)
            lblOutput.config(text=msg, fg="red")
        except Exception as e:
            lblOutput.config(text="Error! Couldn't perform the testing.", fg="red")
            print(e)  # Debug


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def backButton():
        navigateFunction("start")

    def testButton():
        ui.performTesting(lblOutput)

    frame = tk.Frame(window, bg=BACKGROUND_COLOR)
    ui = FrameTestUI()
    tk.Label(frame, text="Model Testing", font=(TITLE_FONT, 24), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).pack()

    # Initialize the widgets
    ui.inputComponents(frame)

    tk.Button(frame, text="Perform Testing!", relief="groove", font=(BUTTON_TEXT, 20), command=testButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).pack(pady=(20, 0))
    lblOutput = tk.Label(frame, text="", bg=BACKGROUND_COLOR)
    lblOutput.pack()
    tk.Button(frame, text="Back to Main Menu", relief="groove", font=(BUTTON_TEXT, 12), command=backButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).pack(side="bottom", pady=10)

    return frame
