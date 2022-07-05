""" Model creation page.
    In this page, the user will be able to load a dataset, configure the pre-processing settings and train the model.
    Also, he will be able to save the trained model for future use.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from preprocessing import findColumns


def settingsFrame(window):
    """ Initialize the pre-processing frame of the page. """
    def binsTextLimit(txt):
        """ Limit the input from the user to be digits only and maximum of 2 digits. """
        if len(txt.get()) > 2:
            txt.set(txt.get()[:2])
        if txt.get() != "" and txt.get()[-1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            txt.set(txt.get()[:-1])

    norm = True

    frame = tk.Frame(window, highlightbackground="black", highlightthickness=1)
    tk.Label(frame, text="Pre-Processing Settings", font=(None, 12)).grid(column=0, row=0, columnspan=2, pady=(15, 5))

    # Initialize the settings options widgets
    lblMissing = tk.Label(frame, text="Missing Values According to:")
    comboMissing = ttk.Combobox(frame, values=["Target Class", "Whole Dataset"], state="readonly", width=15)
    comboMissing.set("Target Class")

    lblNorm = tk.Label(frame, text="Normalization:")
    chkbtnNorm = tk.Checkbutton(frame, text="Apply", variable=norm, onvalue=True, offvalue=False)
    chkbtnNorm.select()

    lblDisc = tk.Label(frame, text="Discretization:")
    comboDisc = ttk.Combobox(frame, values=["No Discretization", "Equal-Width", "Equal-Frequency", "Entropy Based"], state="readonly", width=15)
    comboDisc.set("No Discretization")

    inputBins = tk.StringVar()
    inputBins.trace("w", lambda *args: binsTextLimit(inputBins))
    lblBins = tk.Label(frame, text="Number of Bins:")
    txtBins = tk.Entry(frame, relief="groove", width=3, bd=2, textvariable=inputBins)

    lblRatio = tk.Label(frame, text="Train-Test Split Ratio:")
    comboRatio = ttk.Combobox(frame, values=["90% - 10%", "75% - 25%", "70% - 30%", "50% - 50%"],
                             state="readonly", width=15)
    comboRatio.set("75% - 25%")

    # Placing the widgets on the grid
    lblMissing.grid(column=0, row=1, sticky="w")
    comboMissing.grid(column=1, row=1, sticky="w", padx=5)
    lblNorm.grid(column=0, row=2, sticky="w")
    chkbtnNorm.grid(column=1, row=2, sticky="w", padx=5)
    lblDisc.grid(column=0, row=3, sticky="w")
    comboDisc.grid(column=1, row=3, sticky="w", padx=5)
    lblBins.grid(column=0, row=4)
    txtBins.grid(column=1, row=4, sticky="w", padx=5)
    tk.Label(frame, text="").grid(column=0, row=5) # Filler lines
    lblRatio.grid(column=0, row=6, sticky="w")
    comboRatio.grid(column=1, row=6, sticky="w", padx=5)
    tk.Label(frame, text="").grid(column=0, row=7) # Filler lines

    return frame


def createModelFrame(window):
    """ Initialize the model creation frame of the page. """
    def pathSelect():
        dataPath = fd.askopenfilename(filetypes=(('CSV file', '*.csv'),))
        if dataPath != "":
            columns = findColumns(dataPath)
            comboTarget['values'] = columns
            comboTarget.set("Choose Target Feature..")

    def folderSelect():
        folderPath = fd.askdirectory()
        if folderPath == "":
            lblFolderResult.config(text="**Choose a folder**")
        else:
            lblFolderResult.config(text="Folder Found!")

    frame = tk.Frame(window)
    tk.Label(frame, text="Model Creation", font=(None, 12)).grid(column=0, row=0, columnspan=3, pady=(15, 5))

    # Initialize the widgets
    lblLoadData = tk.Label(frame, text="Load a dataset file:")
    btnLoadData = tk.Button(frame, text="Browse", command=pathSelect, relief="groove")
    comboTarget = ttk.Combobox(frame, state="readonly", width=30)

    lblFolder = tk.Label(frame, text="Where to save the model? ")
    btnFolder = tk.Button(frame, text="Browse", command=folderSelect, relief="groove")
    lblFolderResult = tk.Label(frame, text="**Choose a folder**")

    lblModel = tk.Label(frame, text="Choose a Model:")
    comboModel = ttk.Combobox(frame, values=["Our Naive Bayes", "Our Decision Tree",
                                             "Sklearn's Naive Bayes", "Sklearn's Decision Tree",
                                             "Sklearn's KNN", "Sklearn's K-Means"],
                              state="readonly", width=25)
    comboModel.set("Our Naive Bayes")

    # Placing the widgets on the grid
    lblLoadData.grid(column=0, row=1, sticky="w")
    btnLoadData.grid(column=1, row=1, sticky="w", padx=(5, 0))
    comboTarget.grid(column=2, row=1, sticky="w", padx=(5, 0))
    lblFolder.grid(column=0, row=2, sticky="w")
    btnFolder.grid(column=1, row=2, sticky="w", padx=(5, 0))
    lblFolderResult.grid(column=2, row=2, sticky="w", padx=(5, 0))
    tk.Label(frame, text="").grid(column=0, row=3)  # Filler lines
    lblModel.grid(column=0, row=4, sticky="w")
    comboModel.grid(column=1, row=4, sticky="w", padx=(5, 0), columnspan=2)

    return frame


def getFrame(window):
    """ Initialize the main frame. """
    frame = tk.Frame(window)

    # Initialize the frame's properties
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=7)

    settingsFrame(frame).grid(column=0, row=0, sticky="n")
    createModelFrame(frame).grid(column=1, row=0, sticky="n")
    tk.Button(frame, text="Create and Train the Model!", font=(None, 14)).grid(column=0, row=1, columnspan=2, pady=(20, 0))

    return frame
