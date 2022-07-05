""" Implementing the ID3 decision tree algorithm.

    Authors:  Din Ezra      208273094
              Lior Swissa   318657384
"""
import pandas as pd
import numpy as np
from math import log2
from Models.BaseModel import BaseModel


def calculateEntropy(data, target):
    """ Calculate the entropy of the dataframe for the target class.

    @:param data         DataFrame object.
    @:param target       The target class.

    @:return             Entropy of the dataframe.
    """
    entropy = 0

    for v in data[target].unique():
        freq = data[data[target] == v].shape[0] / data.shape[0]
        entropy -= freq * log2(freq)

    return entropy


def calcInformationGain(data, feature, target):
    """ Calculate the information gain for that specific feature.

    @:param data         DataFrame object.
    @:param feature      The feature for calculating its information gain.
    @:param target       The target class.

    @:return             The information gain of the given feature.
    """
    featureInfo = 0

    for v in data[feature].unique():
        tmpData = data[data[feature] == v]
        featureInfo += (tmpData.shape[0] / data.shape[0]) * calculateEntropy(tmpData, target)

    return calculateEntropy(data, target) - featureInfo


def getBestFeature(data, target):
    """ Choose the feature with the highest information gain.

    @:param data         DataFrame object.
    @:param target       The target class.

    @:return             The best feature, as string.
    """
    maxIG = -1  # Value of the highest information gain
    result = None  # The feature who has the highest information gain
    features = data.columns.drop(target)  # List of all the features in the dataframe, without the target class

    for f in features:
        ig = calcInformationGain(data, f, target)

        if ig > maxIG:
            maxIG = ig
            result = f

    return result


class OurTree(BaseModel):
    def __init__(self):
        """ Initialize the model's object. """
        # Create an empty dictionary - representing the ID3 decision tree.
        self.tree = {} # The tree will be updated upon training
        self.target = None

    def train(self, df, target: str):
        """ Train the new model according to the desired class, and save the trained
        model for later prediction use.

        @:param df         The dataframe for training.
        @:param target     The class attribute for the classification.
        """

        def generateNode(data, feature, target):
            """ Generate a tree node for the given feature in the dataset.

            @:param data         DataFrame object.
            @:param feature      The feature that this node represents.
            @:param target       The target class.

            @:return             A tree node.
            """
            feature_value_count_dict = data[feature].value_counts(sort=False)
            tree = dict()

            for feature_value, count in feature_value_count_dict.iteritems():
                feature_value_data = data[data[feature] == feature_value]

                assigned_to_node = False  # Indicates whether the node is a leaf
                for c in data[target].unique():
                    class_count = feature_value_data[feature_value_data[target] == c].shape[0]

                    if class_count == count:
                        tree[feature_value] = c
                        data = data[data[feature] != feature_value]
                        assigned_to_node = True

                if not assigned_to_node:
                    tree[feature_value] = "?"

            return tree, data

        def createTree(root, prev_feature_value, data, label):
            if data.shape[0] != 0:  # if dataset becomes enpty after updating
                max_info_feature = getBestFeature(data, label)  # most informative feature
                tree, train_data = generateNode(data, max_info_feature, label)  # getting tree node and updated dataset
                next_root = None

                if prev_feature_value != None:  # add to intermediate node of the tree
                    root[prev_feature_value] = dict()
                    root[prev_feature_value][max_info_feature] = tree
                    next_root = root[prev_feature_value][max_info_feature]
                else:  # add to root of the tree
                    root[max_info_feature] = tree
                    next_root = root[max_info_feature]

                for node, branch in list(next_root.items()):  # iterating the tree node
                    if branch == "?":  # if it is expandable
                        feature_value_data = train_data[
                            train_data[max_info_feature] == node]  # using the updated dataset
                        createTree(next_root, node, feature_value_data, label)  # recursive call with updated dataset

        trainedTree = {}
        createTree(trainedTree, None, df, target)

        self.tree = trainedTree
        self.target = target  # The desired class attribute

    def test(self, df):
        """ Test the model with a test dataset.

        @:param df       The dataframe to test the model.
        """
        if self.target is None:
            print("Error! Please train the model before trying to testing it.")
            return

        success = 0
        failed = 0

        for _, row in df.iterrows():
            answer = row[self.target]
            prediction = self.predict(row.drop(labels=[self.target]))

            if answer == prediction:
                success += 1
            else:
                failed += 1

        print("Success:\t", success)
        print("Failed:\t\t", failed)
        print("Accuracy Rate: ", (success / (success + failed)) * 100, "%")

    def predict(self, query):
        def traverse(tree, instance):
            if not isinstance(tree, dict):
                return tree  # We reached a leaf node - return it
            else:
                root_node = next(iter(tree))  # Get the first key of the dictionary
                feature_value = instance[root_node]

                if feature_value in tree[root_node]:  # Check if the feature value is in current tree node
                    return traverse(tree[root_node][feature_value], instance)  # Traverse the tree
                else:
                    return None

        return traverse(self.tree, query)