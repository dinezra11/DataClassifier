import tkinter as tk

import UI.frameCreate


def drawTitle(window):
    tk.Label(window, text="Data Mining - Final Project", font=(None, 32)).pack(pady=(10, 0))
    tk.Label(window, text="Developed by Din Ezra and Lior Swissa", font=(None, 10)).pack()


def main():
    # Create new window and calculate the screen's center coordinates
    window = tk.Tk()
    windowSize = (800, 500)
    windowPos = (int(window.winfo_screenwidth() / 2 - windowSize[0] / 2),
                 int(window.winfo_screenheight() / 2 - windowSize[1] / 2))
    windowTitle = "Data Mining Final Project"

    # Initialize project's window properties
    window.title(windowTitle)
    window.geometry(f'{windowSize[0]}x{windowSize[1]}+{windowPos[0]}+{windowPos[1]}')
    window.resizable(False, False)

    # Draw project's title
    drawTitle(window)

    # Set current active frame:
    UI.frameCreate.getFrame(window).pack(fill="both")

    window.mainloop()


if __name__ == "__main__":
    main()
