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
from Models.OurNaiveBayes import OurNaiveBayes
from Models.OurDecisionTree import OurTree
from Models.NaiveBayes import NaiveBayes
from Models.DecisionTree import DecisionTree


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

        frame = tk.Frame(window, highlightbackground="black", highlightthickness=1)
        tk.Label(frame, text="Pre-Processing Settings", font=(None, 12)).grid(column=0, row=0, columnspan=2,
                                                                              pady=(15, 5))

        # Initialize the settings options widgets
        lblMissing = tk.Label(frame, text="Missing Values According to:")
        self.comboMissing = ttk.Combobox(frame, values=["Target Class", "Whole Dataset"], state="readonly", width=15)
        self.comboMissing.set("Target Class")

        lblNorm = tk.Label(frame, text="Normalization:")
        chkbtnNorm = tk.Checkbutton(frame, text="Apply", variable=self.norm, onvalue=True, offvalue=False)
        chkbtnNorm.select()

        lblDisc = tk.Label(frame, text="Discretization:")
        self.comboDisc = ttk.Combobox(frame,
                                      values=["No Discretization", "Equal-Width", "Equal-Frequency", "Entropy Based"],
                                      state="readonly", width=15)
        self.comboDisc.set("No Discretization")

        inputBins = tk.StringVar()
        inputBins.trace("w", lambda *args: binsTextLimit(inputBins))
        lblBins = tk.Label(frame, text="Number of Bins:")
        self.txtBins = tk.Entry(frame, relief="groove", width=3, bd=2, textvariable=inputBins)

        lblRatio = tk.Label(frame, text="Train-Test Split Ratio:")
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
        tk.Label(frame, text="").grid(column=0, row=5)  # Filler lines
        lblRatio.grid(column=0, row=6, sticky="w")
        self.comboRatio.grid(column=1, row=6, sticky="w", padx=5)
        tk.Label(frame, text="").grid(column=0, row=7)  # Filler lines

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

        def folderSelect():
            self.folderPath = fd.askdirectory()
            if self.folderPath == "":
                lblFolderResult.config(text="**Choose a folder**")
                self.folderPath = None
            else:
                lblFolderResult.config(text="Folder Found!")

        frame = tk.Frame(window)
        tk.Label(frame, text="Model Creation", font=(None, 12)).grid(column=0, row=0, columnspan=3, pady=(15, 5))

        # Initialize the widgets
        lblLoadData = tk.Label(frame, text="Load a dataset file:")
        btnLoadData = tk.Button(frame, text="Browse", command=pathSelect, relief="groove")
        self.comboTarget = ttk.Combobox(frame, state="readonly", width=30)

        lblFolder = tk.Label(frame, text="Where to save the model? ")
        btnFolder = tk.Button(frame, text="Browse", command=folderSelect, relief="groove")
        lblFolderResult = tk.Label(frame, text="**Choose a folder**")

        lblModel = tk.Label(frame, text="Choose a Model:")
        self.comboModel = ttk.Combobox(frame, values=["Our Naive Bayes", "Our Decision Tree",
                                                      "Sklearn's Naive Bayes", "Sklearn's Decision Tree",
                                                      "Sklearn's KNN", "Sklearn's K-Means"],
                                       state="readonly", width=25)
        self.comboModel.set("Our Naive Bayes")

        # Placing the widgets on the grid
        lblLoadData.grid(column=0, row=1, sticky="w")
        btnLoadData.grid(column=1, row=1, sticky="w", padx=(5, 0))
        self.comboTarget.grid(column=2, row=1, sticky="w", padx=(5, 0))
        lblFolder.grid(column=0, row=2, sticky="w")
        btnFolder.grid(column=1, row=2, sticky="w", padx=(5, 0))
        lblFolderResult.grid(column=2, row=2, sticky="w", padx=(5, 0))
        tk.Label(frame, text="").grid(column=0, row=3)  # Filler lines
        lblModel.grid(column=0, row=4, sticky="w")
        self.comboModel.grid(column=1, row=4, sticky="w", padx=(5, 0), columnspan=2)

        return frame

    def outputFrame(self, window):
        """ Initialize the output section of the page.
        Successful outputs and error messages will appear to the user here.

        @:param window              The main frame of the page.
        @:return                    The sub-frame as the output.
        """
        frame = tk.Frame(window)

        self.lblOutput = tk.Label(frame, text="Output:")

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

            # Load the dataset
            self.setOutput("Loading dataset..")
            data = loadCsv(self.dataPath)

            # Perform preprocessing
            self.setOutput("Performing data pre-processing..")
            trainSet, testSet = dataPreprocess(data, input["target"], input["missing"], input["norm"], input["disc"], input["splitRatio"], input["folderPath"])

            # Model Training
            self.setOutput("Creating and training the model..")
            model = input["model"]()
            model.train(trainSet, input["target"])
            model.save(str(input["folderPath"]) + "/model")

            self.setOutput("Model has been created, trained and saved. You can go test it now! :)", 0)
        except ValueError as ve:
            self.setOutput(ve.__str__(), -1)
        except Exception:
            self.setOutput("Unexpected error occurred!")


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def backButton():
        navigateFunction("main")

    frame = tk.Frame(window)
    userInterface = FrameCreateUI()

    # Initialize the frame's properties
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=7)

    userInterface.settingsFrame(frame).grid(column=0, row=0, sticky="n")
    userInterface.createModelFrame(frame).grid(column=1, row=0, sticky="n")
    tk.Button(frame, text="Create and Train the Model!", relief="groove", font=(None, 14),
              command=userInterface.createModel).grid(column=0, row=1, columnspan=2, pady=(20, 0))
    userInterface.outputFrame(frame).grid(column=0, row=2, columnspan=2, sticky="w", pady=(35, 0))
    tk.Button(frame, text="Back to Main Menu", font=(None, 12), command=backButton).grid(column=0, row=3, columnspan=2, sticky="s", pady=10)

    return frame
