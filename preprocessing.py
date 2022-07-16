""" Useful functions for data pre-processing.
    The function "dataPreprocess()" calls all of the other functions and perform a complete pre-processing pipeline.
    The rest of the functions perform a specific pre-processing related task.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


def splitData(data: pd.DataFrame, target: str, ratio):
    """ Split the dataframe to train/test files.

    @:param data            Dataframe for splitting.
    @:param target          The target class feature.
    @:param ratio           The train/test ratio when splitting (in percentages).

    @:return                2 dataframe objects, one for train set and one for test set.
                            Both dataframes are splitted from the original given dataframe.
    """
    if not 50 <= ratio < 100:
        raise ValueError("Error, invalid input for ratio.")
    if target not in data.columns:
        raise ValueError("Error, the target selected is not part of the data's features")

    train, test = tts(data, train_size=ratio / 100)

    return train, test


def saveCsv(data: pd.DataFrame, path: str, filename: str):
    """ Save the dataframe into a csv file.

    @:param data            Dataframe to save.
    @:param path            The path to the folder where to save the file.
    @:param filename        The filename of the new csv file.
    """
    data.to_csv(path + "/" + filename + "_clean.csv", index=False)


def fillMissing(data: pd.DataFrame, target: str = None):
    """ Fill the missing cells of the dataframe with a value.
    For numeric features - fill with the average of the column.
    For categorical features - fill with the most frequent value of the column.

    @:param data        Dataframe.
    @:param target      The target feature. The missing cells will be filled according to the target's values.
                        If None - the missing cells will be filled according to the whole dataset.
    """
    # Fill missing in numeric columns
    numeric = data.select_dtypes(include=np.number).columns
    if not numeric.empty:
        if target is None:
            data[numeric] = data[numeric].fillna(data.mean(numeric_only=True))
        else:
            data[numeric] = data[numeric].fillna(data.groupby(target)[numeric].transform("mean"))

    # Fill missing in categorical columns
    category = data.select_dtypes(exclude=np.number).columns
    if not category.empty:
        if target is None:
            data[category] = data[category].fillna(data.mode().iloc[0])
        else:
            groups = data.groupby(target)
            mode_by_group = groups[category].transform(lambda x: x.mode()[0])
            data[category] = data[category].fillna(mode_by_group)


def normalization(data: pd.DataFrame):
    """ Perform a normalization on the data.

    @:param data            Dataframe.
    """
    scaler = StandardScaler()
    columns = data.select_dtypes(include=np.number).columns  # Get numeric columns for discretization

    for c in columns:
        data[c] = scaler.fit_transform(data[c].values.reshape(-1, 1))


def discretization(data: pd.DataFrame, discType: int, binsNumber: int):
    """ Perform discretization on the data.

    @:param data            Dataframe.
    @:param discType        The type of the discretization:
                            0 - Based equal-width.
                            1 - Based equal-frequency.
                            2 - Based entropy.
    @:param binsNumber      The number of bins.
    """
    if (not 0 <= discType <= 2) or (binsNumber < 0):
        raise ValueError("Error, invalid input for discretization method.")

    columns = data.select_dtypes(include=np.number).columns # Get numeric columns for discretization

    for col in columns:
        if discType == 0: # Discretization based equal-width.
            data[col] = pd.cut(x=data[col], bins=binsNumber)
        elif discType == 1: # Discretization based equal-frequency.
            data[col] = pd.qcut(x=data.rank(method="first")[col], q=binsNumber)
        else: # Discretization based entropy.
            data[col] = pd.cut(x=data[col], bins=binsNumber)

        data[col] = data[col].astype("category") # Convert the type to discrete type


def findColumns(path: str):
    """ Load a data file, and return a list of its columns (features).

    @:param path            The path to the data file.

    @:return                A list of all of the data's columns.
    """
    try:
        data = pd.read_csv(path)
        columns = list(data.columns)
        return columns
    except Exception:
        raise ValueError("Error! Data path is invalid.")


def loadCsv(path: str):
    """ Load a csv file.

    @:param path            The path to the csv file.
    @:return                Panda's DataFrame.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        raise ValueError("Error! Invalid path for data.")


def dataPreprocess(data: pd.DataFrame, target: str, fillMissingCls: bool, normalize: bool, discretizeType: tuple, splitRatio: int, folderPath: str):
    """ Perform a complete pre-processing over the given data.
    Call the above functions to help doing the job.

    @:param data            Dataframe for pre-processing.
    @:param target          The target class feature.
    @:param fillMissingCls  Boolean, true for filling the missing cells according to the target values.
                            False for filling the missing cells according to the whole dataset.
    @:param normalize       Boolean, indicates whether a normalization needs to be performed.
    @:param discretizeType  A tuple for the discretization's settings. -> (type, bins_n)
                            Where "bins_n" is integer, indicating the number of bins:
                                (0, bins_n) -> Based equal-width.
                                (1, bins_n) -> Based equal-frequency.
                                (2, bins_n) -> Based entropy.
                                None -> No discretization
    @:param splitRation     The ratio for train-test sets splitting.
    @:param folderPath      The path to the folder where to save the clean csv files.

    @:return                The data after pre-processing.
                            The data is returned as a train/test splitted sets.
    """
    # Drop rows with empty target value
    data.dropna(subset=[target], inplace=True)
    if is_numeric_dtype(data[target]):
        data[target] = data[target].astype("category")

    # Fill missing values (according to target feature only or not)
    if fillMissingCls:
        fillMissing(data, target)
    else:
        fillMissing(data)

    # Perform normalization on the data if needed
    if normalize:
        normalization(data)

    # Perform discretization on the data if needed
    if type(discretizeType) is tuple:
        discretization(data, discretizeType[0], discretizeType[1])

    # Split the data into train/test sets
    train, test = splitData(data, target, splitRatio)

    # Save clean data into csv files and return them as dataframe objects
    saveCsv(train, folderPath, "train")
    saveCsv(test, folderPath, "test")

    return train, test
