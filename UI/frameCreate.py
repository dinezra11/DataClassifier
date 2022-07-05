""" Model creation page.
    In this page, the user will be able to load a dataset, configure the pre-processing settings and train the model.
    Also, he will be able to save the trained model for future use.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk
from tkinter import ttk


def settingsFrame(window):
    """ Initialize the pre-processing frame of the page. """
    def binsTextLimit(txt):
        """ Limit the input from the user to be digits only and maximum of 2 digits. """
        if len(txt.get()) > 2:
            txt.set(txt.get()[:2])
        if txt.get() != "" and txt.get()[-1] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            txt.set(txt.get()[:-1])

    norm = True

    frame = tk.Frame(window)
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

    # Placing the widgets on the grid
    lblMissing.grid(column=0, row=1, sticky="w")
    comboMissing.grid(column=1, row=1, sticky="w", padx=(5, 0))
    lblNorm.grid(column=0, row=2, sticky="w")
    chkbtnNorm.grid(column=1, row=2, sticky="w", padx=(5, 0))
    lblDisc.grid(column=0, row=3, sticky="w")
    comboDisc.grid(column=1, row=3, sticky="w", padx=(5, 0))
    lblBins.grid(column=0, row=4,)
    txtBins.grid(column=1, row=4, sticky="w", padx=(5, 0))

    return frame


def getFrame(window):
    frame = tk.Frame(window)

    # Initialize the frame's properties
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=5)

    settingsFrame(frame).grid(column=0, row=0)

    return frame
