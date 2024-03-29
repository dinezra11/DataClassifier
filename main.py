""" Main script!
    To run the system, run this file. The main function is included here.

    This file is in charge of initializing the system's window, showing the current page to the user and perform
    navigation between different pages.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import tkinter as tk

import UI.frameStart
import UI.frameCreate
import UI.frameTest
import UI.frameAbout
from UI.colors import *


def drawTitle(window):
    tk.Label(window, text="Data Mining - Final Project", bg=BACKGROUND_COLOR, fg=TITLE_COLOR, font=(TITLE_FONT, 32)).pack(pady=(10, 0))
    tk.Label(window, text="Developed by Din Ezra and Lior Swissa", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(TEXT_FONT, 10)).pack(pady=(0, 5))


def main():
    """ Main function.
    Initialize the project's window and perform navigation across the pages.
    """

    def navigate(page):
        """ Navigate the program to another page.

        @:param page            The name of the new page.
                                Should be one of the constants declared in this file.
        """
        nonlocal current

        if type(current) is tk.Frame:
            current.pack_forget()
            current.destroy()

        if page == "start":
            current = UI.frameStart.getFrame(window, navigate)
        elif page == "create":
            current = UI.frameCreate.getFrame(window, navigate)
        elif page == "test":
            current = UI.frameTest.getFrame(window, navigate)
        else:
            current = UI.frameAbout.getFrame(window, navigate)

        current.pack(expand=1, fill="both")

    # Create new window and calculate the screen's center coordinates
    window = tk.Tk()
    windowSize = (800, 500)
    windowPos = (int(window.winfo_screenwidth() / 2 - windowSize[0] / 2),
                 int(window.winfo_screenheight() / 2 - windowSize[1] / 2))
    windowTitle = "Data Mining Final Project"
    window.config(background=BACKGROUND_COLOR)

    # Initialize project's window properties
    window.title(windowTitle)
    window.geometry(f'{windowSize[0]}x{windowSize[1]}+{windowPos[0]}+{windowPos[1]}')
    window.resizable(False, False)

    # Draw project's title
    drawTitle(window)

    # Set current active page:
    current = None
    navigate("start")

    window.mainloop()


if __name__ == "__main__":
    main()
