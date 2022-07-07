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

    def aboutButton():
        navigateFunction("about")

    frame = tk.Frame(window)

    # Initialize the frame's widgets
    tk.Label(frame, text="About Us", font=(None, 24)).pack()
    tk.Label(frame, text='This project is part of the course "Intro to Data Mining" in SCE.').pack(anchor="w")

    # Menu
    tk.Button(frame, text="Create a Model", relief="groove", font=(None, 12), command=createButton).pack()
    tk.Button(frame, text="About Us", relief="groove", font=(None, 12), command=aboutButton).pack()

    return frame
