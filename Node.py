import pandas as pd
import numpy as np
from collections import Counter


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (str): The feature used for splitting at this node.
        threshold (float): The threshold value used for splitting at this node.
        children (dict): A dictionary of child nodes, where the keys are the possible feature values and the values are the corresponding child nodes.
        value (float): The predicted value at this node (only applicable for leaf nodes).
    """

    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children if children is not None else {}
        self.value = value

    def isLeaf(self):
        return self.value is not None
