""" Model creation page.
    In this page, the user will be able to load a dataset, configure the pre-processing settings and train the model.
    Also, he will be able to save the trained model for future use.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk

from preprocessing import findColumns, loadCsv, dataPreprocess
from Models.ourNaiveBayes import OurNaiveBayes
from Models.ourDecisionTree import OurTree
from Models.naiveBayes import NaiveBayes
from Models.decisionTree import DecisionTree
from Models.kNeighbors import KNeighbors
from Models.kMeans import KMeansModel
from UI.colors import *


class FrameCreateUI:
    """ A wrapper class for the different frames of the model creation page.
    We use a class here for convenient development of the frames and for a better encapsulated code.
    """

    def __init__(self):
        # Settings Variables
        self.norm = True
        self.comboMissing = None
        self.comboDisc = None
        self.txtBins = None
        self.comboRatio = None

        # Model Creation Variables
        self.dataPath = None
        self.folderPath = None
        self.comboTarget = None
        self.comboModel = None
        self.txtK = None

        # Output/Error Variables
        self.lblOutput = None

    def settingsFrame(self, window):
        """ Initialize the pre-processing frame of the page.

        @:param window              The main frame of the page.
        @:return                    The sub-frame of the pre-processing settings.
        """

        def binsTextLimit(txt):
            """ Limit the input from the user to be digits only and maximum of 2 digits. """
            if len(txt.get()) > 2:
                txt.set(txt.get()[:2])
            if txt.get() != "" and txt.get()[-1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                txt.set(txt.get()[:-1])

        self.norm = True

        frame = tk.Frame(window, highlightbackground="green", highlightthickness=1, bg=BACKGROUND_COLOR)
        tk.Label(frame, text="Pre-Processing Settings", font=(TITLE_FONT, 12), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).grid(column=0, row=0, columnspan=2,
                                                                              pady=(15, 5))

        # Initialize the settings options widgets
        lblMissing = tk.Label(frame, text="Missing Values According to:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.comboMissing = ttk.Combobox(frame, values=["Target Class", "Whole Dataset"], state="readonly", width=15)
        self.comboMissing.set("Target Class")

        lblNorm = tk.Label(frame, text="Normalization:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        chkbtnNorm = tk.Checkbutton(frame, text="Apply", variable=self.norm, onvalue=True, offvalue=False)
        chkbtnNorm.select()

        lblDisc = tk.Label(frame, text="Discretization:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.comboDisc = ttk.Combobox(frame,
                                      values=["No Discretization", "Equal-Width", "Equal-Frequency", "Entropy Based"],
                                      state="readonly", width=15)
        self.comboDisc.set("No Discretization")

        inputBins = tk.StringVar()
        inputBins.trace("w", lambda *args: binsTextLimit(inputBins))
        lblBins = tk.Label(frame, text="Number of Bins:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.txtBins = tk.Entry(frame, relief="groove", width=3, bd=2, textvariable=inputBins)

        lblRatio = tk.Label(frame, text="Train-Test Split Ratio:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.comboRatio = ttk.Combobox(frame, values=["90% - 10%", "75% - 25%", "70% - 30%", "50% - 50%"],
                                       state="readonly", width=15)
        self.comboRatio.set("75% - 25%")

        # Placing the widgets on the grid
        lblMissing.grid(column=0, row=1, sticky="w")
        self.comboMissing.grid(column=1, row=1, sticky="w", padx=5)
        lblNorm.grid(column=0, row=2, sticky="w")
        chkbtnNorm.grid(column=1, row=2, sticky="w", padx=5)
        lblDisc.grid(column=0, row=3, sticky="w")
        self.comboDisc.grid(column=1, row=3, sticky="w", padx=5)
        lblBins.grid(column=0, row=4)
        self.txtBins.grid(column=1, row=4, sticky="w", padx=5)
        tk.Label(frame, text="", bg=BACKGROUND_COLOR).grid(column=0, row=5)  # Filler lines
        lblRatio.grid(column=0, row=6, sticky="w")
        self.comboRatio.grid(column=1, row=6, sticky="w", padx=5)
        tk.Label(frame, text="", bg=BACKGROUND_COLOR).grid(column=0, row=7)  # Filler lines

        return frame

    def createModelFrame(self, window):
        """ Initialize the model creation frame of the page.

        @:param window              The main frame of the page.
        @:return                    The sub-frame of the model creation configurations.
        """

        def pathSelect():
            self.dataPath = fd.askopenfilename(filetypes=(('CSV file', '*.csv'),))
            if self.dataPath != "":
                columns = findColumns(self.dataPath)
                self.comboTarget['values'] = columns
                self.comboTarget.set("Choose Target Feature..")
            else:
                self.dataPath = None
                columns = None
                self.comboTarget['values'] = columns
                self.comboTarget.set("")

        def folderSelect():
            self.folderPath = fd.askdirectory()
            if self.folderPath == "":
                lblFolderResult.config(text="**Choose a folder**")
                self.folderPath = None
            else:
                lblFolderResult.config(text="Folder Found!")

        def textLimit(txt):
            """ Limit the input from the user to be digits only and maximum of 2 digits. """
            if len(txt.get()) > 2:
                txt.set(txt.get()[:2])
            if txt.get() != "" and txt.get()[-1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                txt.set(txt.get()[:-1])

        frame = tk.Frame(window, bg=BACKGROUND_COLOR)
        tk.Label(frame, text="Model Creation", font=(TITLE_FONT, 12), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).grid(column=0, row=0, columnspan=3, pady=(15, 5))

        # Classification Frame
        lblLoadData = tk.Label(frame, text="Load a dataset file:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        btnLoadData = tk.Button(frame, text="Browse", command=pathSelect, relief="groove", bg=BUTTON_BACK, fg=BUTTON_TEXT)
        self.comboTarget = ttk.Combobox(frame, state="readonly", width=30)

        lblFolder = tk.Label(frame, text="Where to save the model? ", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        btnFolder = tk.Button(frame, text="Browse", command=folderSelect, relief="groove", bg=BUTTON_BACK, fg=BUTTON_TEXT)
        lblFolderResult = tk.Label(frame, text="**Choose a folder**", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)

        lblModel = tk.Label(frame, text="Choose a Model:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.comboModel = ttk.Combobox(frame, values=["Our Naive Bayes", "Our Decision Tree",
                                                      "Sklearn's Naive Bayes", "Sklearn's Decision Tree",
                                                      "Sklearn's KNN", "Sklearn's K-Means"],
                                       state="readonly", width=25)
        self.comboModel.set("Our Naive Bayes")

        inputK = tk.StringVar()
        inputK.trace("w", lambda *args: textLimit(inputK))
        lblK = tk.Label(frame, text="K (neighbors/clusters):", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        self.txtK = tk.Entry(frame, relief="groove", width=3, bd=2, textvariable=inputK)

        # Placing the widgets on the grid
        lblLoadData.grid(column=0, row=1, sticky="w")
        btnLoadData.grid(column=1, row=1, sticky="w", padx=(5, 0))
        self.comboTarget.grid(column=2, row=1, sticky="w", padx=(5, 0))
        lblFolder.grid(column=0, row=2, sticky="w")
        btnFolder.grid(column=1, row=2, sticky="w", padx=(5, 0))
        lblFolderResult.grid(column=2, row=2, sticky="w", padx=(5, 0))
        tk.Label(frame, text="", bg=BACKGROUND_COLOR).grid(column=0, row=3)  # Filler lines
        lblModel.grid(column=0, row=4, sticky="w")
        self.comboModel.grid(column=1, row=4, sticky="w", padx=(5, 0), columnspan=2)
        lblK.grid(column=0, row=5, sticky="w", pady=(10, 0))
        self.txtK.grid(column=1, row=5, sticky="w", padx=(5, 0), pady=(10, 0))

        return frame

    def outputFrame(self, window):
        """ Initialize the output section of the page.
        Successful outputs and error messages will appear to the user here.

        @:param window              The main frame of the page.
        @:return                    The sub-frame as the output.
        """
        frame = tk.Frame(window, bg=BACKGROUND_COLOR)

        self.lblOutput = tk.Label(frame, text="Output:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR)

        # Placing the widgets on the grid
        self.lblOutput.pack()

        return frame

    def setOutput(self, txt: str, type: int = None):
        """ Set the output to the user.

        @:param txt         The message for the output.
        @:param type        Message's type as integer.
                            0       -> Successful       (green text)
                            -1      -> Error            (red text)
                            other   -> Regular          (black text)
        """
        if type == 0:
            type = "green"
        elif type == -1:
            type = "red"
        else:
            type = "black"

        txt = "Output: " + txt
        self.lblOutput.config(text=txt, fg=type)

    def validateWidgets(self):
        """ Validate the legitimate of the widgets. (Check all of them already been initialized).

        @:return            Boolean value, indicates whether all of the widgets are fully initialized.
        """
        if self.comboMissing is None or self.comboDisc is None or self.txtBins is None or self.comboRatio is None or \
                self.dataPath is None or self.folderPath is None or self.comboTarget is None or self.comboModel is None or \
                self.comboTarget.get() == "Choose Target Feature..":
            return False

        return True

    def createModel(self):
        """ Create, train and save the desired model according to the input settings! """
        try:
            self.setOutput("Performing data pre-processing..")

            # Validate settings
            if not self.validateWidgets():
                raise ValueError("Please define all of the settings and configurations before creating to model.")

            # Getting input from widgets
            input = {
                "missing": self.comboMissing.get() == "Target Class",
                "norm": self.norm,
                "splitRatio": int(self.comboRatio.get()[:2]),
                "target": self.comboTarget.get(),
                "folderPath": self.folderPath
            }
            if self.comboDisc.get() == "Equal-Width":
                input["disc"] = (0, int(self.txtBins.get()))
            elif self.comboDisc.get() == "Equal-Frequency":
                input["disc"] = (1, int(self.txtBins.get()))
            elif self.comboDisc.get() == "Entropy Based":
                input["disc"] = (2, int(self.txtBins.get()))
            else:
                input["disc"] = None

            if self.comboModel.get() == "Our Naive Bayes":
                input["model"] = OurNaiveBayes
            elif self.comboModel.get() == "Our Decision Tree":
                input["model"] = OurTree
            elif self.comboModel.get() == "Sklearn's Naive Bayes":
                input["model"] = NaiveBayes
            elif self.comboModel.get() == "Sklearn's Decision Tree":
                input["model"] = DecisionTree
            else:
                try:
                    input["k"] = int(self.txtK.get())
                except ValueError:
                    raise ValueError("Please specify value of 'k'.")
                else:
                    if self.comboModel.get() == "Sklearn's KNN":
                        input["model"] = KNeighbors
                    else:
                        input["model"] = KMeansModel

            # Load the dataset
            self.setOutput("Loading dataset..")
            data = loadCsv(self.dataPath)

            # Perform preprocessing
            self.setOutput("Performing data pre-processing..")
            trainSet, testSet = dataPreprocess(data, input["target"], input["missing"], input["norm"], input["disc"], input["splitRatio"], input["folderPath"])

            # Model Training
            self.setOutput("Creating and training the model..")
            model = input["model"]()

            if input["model"] == KNeighbors:
                model.train(trainSet, input["target"], input["k"])
            elif input["model"] == KMeansModel:
                model.train(trainSet.drop(columns=[input["target"]]), input["k"])
            else:
                model.train(trainSet, input["target"])

            model.save(str(input["folderPath"]) + "/model")

            self.setOutput("Model has been created, trained and saved. You can go test it now! :)", 0)
        except ValueError as ve:
            self.setOutput(ve.__str__(), -1)
        except Exception as e:
            self.setOutput("Unexpected error occurred!", -1)
            print(e) # Debug


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def backButton():
        navigateFunction("start")

    frame = tk.Frame(window, bg=BACKGROUND_COLOR)
    userInterface = FrameCreateUI()

    # Initialize the frame's properties
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=7)

    userInterface.settingsFrame(frame).grid(column=0, row=0, sticky="n")
    userInterface.createModelFrame(frame).grid(column=1, row=0, sticky="n")
    tk.Button(frame, text="Create and Train the Model!", relief="groove", font=(BUTTON_TEXT, 14),
              command=userInterface.createModel, bg=BUTTON_BACK, fg=BUTTON_TEXT).grid(column=0, row=1, columnspan=2, pady=(20, 0))
    userInterface.outputFrame(frame).grid(column=0, row=2, columnspan=2, sticky="w", pady=(35, 0))
    tk.Button(frame, text="Back to Main Menu", relief="groove", font=(BUTTON_TEXT, 12), command=backButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).grid(column=0, row=3, columnspan=2, sticky="s", pady=10)

    return frame
