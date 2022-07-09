""" Main Menu page.
    This is the intro page of our system.
    This page has a main menu, so the user can navigate to any available page he wants.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk

from UI.colors import *


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def createButton():
        navigateFunction("create")

    def testButton():
        navigateFunction("test")

    def aboutButton():
        navigateFunction("about")

    frame = tk.Frame(window, bg=BACKGROUND_COLOR)

    # Initialize the frame's widgets
    tk.Label(frame, text="Classifier Project", font=(TITLE_FONT, 28), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).pack(pady=(40, 0))

    # Menu
    tk.Label(frame, text="Main Menu", font=(TITLE_FONT, 20), bg=BACKGROUND_COLOR, fg=TITLE_COLOR).pack(pady=(20, 2))
    tk.Button(frame, text="Create a Model", relief="groove", font=(BUTTON_FONT, 12), command=createButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).pack(pady=(10, 2))
    tk.Button(frame, text="Test a Model", relief="groove", font=(BUTTON_FONT, 12), command=testButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).pack(pady=2)
    tk.Button(frame, text="About Us", relief="groove", font=(BUTTON_FONT, 12), command=aboutButton, bg=BUTTON_BACK, fg=BUTTON_TEXT).pack(pady=2)

    tk.Label(frame, text='The project has been developed as part of the "Into to Data Mining" course in SCE.',
             bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(TEXT_FONT, 8)).pack(side="bottom")

    return frame
