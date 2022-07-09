""" About page.
    This page will show some information about the project and about the developers.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk
import webbrowser
from UI.colors import *


def teamInfoFrame(root):
    """ Create the development team's information section. """

    def dinLinkedIn():
        webbrowser.open("https://www.linkedin.com/in/din-ezra-05abb4216/")

    def liorLinkedIn():
        webbrowser.open("https://www.linkedin.com/in/lior-swissa-752884240/")

    frame = tk.Frame(root, bg=BACKGROUND_COLOR)

    # Initialize the frame's properties
    tk.Label(frame, text="Development Team:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=0, row=0, padx=(0, 10))
    tk.Label(frame, text="Din Ezra (dinezra11@gmail.com)", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=0, sticky="w")
    tk.Label(frame, text="Lior Swissa (Liorsw5@gmail.com)", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=1, sticky="w")

    tk.Button(frame, text="Linkedin", command=dinLinkedIn, bg=BUTTON_BACK, fg=BUTTON_TEXT).grid(column=2, row=0, padx=(10, 0))
    tk.Button(frame, text="Linkedin", command=liorLinkedIn, bg=BUTTON_BACK, fg=BUTTON_TEXT).grid(column=2, row=1, padx=(10, 0))

    return frame


def modelsInfoFrame(root):
    """ Create the available models information section. """
    frame = tk.Frame(root, bg=BACKGROUND_COLOR)

    # Initialize the frame's properties
    tk.Label(frame, text="The project include the following machine learning models:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=0, row=0, padx=(0, 10))
    tk.Label(frame, text="Naive Bayes", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=0, sticky="w")
    tk.Label(frame, text="ID3 Decision Tree", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=1, sticky="w")
    tk.Label(frame, text="KNN", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=2, sticky="w")
    tk.Label(frame, text="K-Means", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).grid(column=1, row=3, sticky="w")

    return frame


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def backButton():
        navigateFunction("start")

    frame = tk.Frame(window, bg=BACKGROUND_COLOR)

    # Initialize the frame's widgets
    tk.Label(frame, text="About Us", font=(TITLE_FONT, 24), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).pack()
    tk.Label(frame, text='This project is part of the course "Intro to Data Mining" in SCE.', bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(anchor="w")
    tk.Label(frame, text="The project is a classifier application, which can load a dataset, perform pre-processing "
                         "to it and create a machine learning model from it.", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(anchor="w")
    tk.Label(frame, text="It also tests the new model and provide the accuracy measurements.", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(anchor="w")
    modelsInfoFrame(frame).pack(anchor="w", pady=(20, 0))
    tk.Label(frame, text="The project include our own implementations for the Naive Bayes and ID3 algorithms, "
                         "and alternative implementations from sklearn library.", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(anchor="w")
    teamInfoFrame(frame).pack(anchor="w", pady=(55, 0))

    tk.Button(frame, text="Back to Main Menu", relief="groove", font=(BUTTON_FONT, 12), bg=BUTTON_BACK, fg=BUTTON_TEXT, command=backButton).pack(side="bottom", pady=10)

    return frame
