""" This python file is in charge of saving and loading the trained models.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import pickle


def saveModel(filename: str, model):
    """ Save the trained model for future use.

    @:param filename         The name of the new file to save.
    @:param model            The model's object to be saved
    """
    if model is None:
        print("Error! You need to train the model before saving it.\n")
        return

    try:
        file = open(filename, "wb")
        pickle.dump(model, file)
        file.close()
    except Exception as e:
        raise "Error! Could not save the model.\n"


def loadModel(filepath: str):
    """ Load a model.

    @:param filepath         The path to the saved model to be loaded.
    """
    file = None

    try:
        file = open(filepath, "rb")
        model = pickle.load(file)
        file.close()

        return model
    except Exception as e:
        raise "Error! Could not load the model."
