""" Main Menu page.
    This is the intro page of our system.
    This page has a main menu, so the user can navigate to any available page he wants.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk


def getFrame(window, navigateFunction):
    """ Initialize the main frame. """

    def createButton():
        navigateFunction("create")

    def testButton():
        navigateFunction("test")

    def aboutButton():
        navigateFunction("about")

    frame = tk.Frame(window)

    # Initialize the frame's widgets
    tk.Label(frame, text="Classifier Project", font=(None, 28)).pack(pady=(40, 0))

    # Menu
    tk.Label(frame, text="Main Menu", font=(None, 20)).pack(pady=(20, 2))
    tk.Button(frame, text="Create a Model", relief="groove", font=(None, 12), command=createButton).pack(pady=(10, 2))
    tk.Button(frame, text="Test a Model", relief="groove", font=(None, 12), command=testButton).pack(pady=2)
    tk.Button(frame, text="About Us", relief="groove", font=(None, 12), command=aboutButton).pack(pady=2)

    tk.Label(frame, text='The project has been developed as part of the "Into to Data Mining" course in SCE.').pack(side="bottom")

    return frame
