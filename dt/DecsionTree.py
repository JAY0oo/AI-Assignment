
from Node import Node
import pandas as pd
import numpy as np


class Tree:
    """
    A class representing a decision tree classifier.

    Parameters:
    - maxDepth (int): The maximum depth the decision tree can go. Default = 100.
    - minSampleSplit (int): The minimum number of samples required to split an internal node. Default = 2.
    - nFeature (int): The number of features to consider when looking for the best split. If None, all features will be considered. Default = None.

    Methods:
    - fit(x, y): fit the decision tree to the training data.
    - predict(X): Predict the class labels for the input data.
    """

    def __init__(self, maxDepth=100, minSampleSplit=2, nFeature=None):
        self.maxDepth = maxDepth
        self.minSampleSplit = minSampleSplit
        self.nFeature = nFeature
        self.root = None

    def fit(self, x, y):
        self.nFeature = x.shape[1] if not self.nFeature else min(
            x.shape[1], self.nFeature)

        self.root = self._grow_(x, y)

    def predict(self, X):
        return np.array([self._traverse_(x, self.root) for x in X])

    def _traverse_(self, x, node):
        # helper function to traverse the tree

        if node.isLeaf():
            return node.value

        child = node.children.get(x[node.feature])

        if child:
            return self._traverse_(x, child)

        return None

    def _grow_(self, x, y, depth=0):
        # helper function to grow the tree

        numSamples, numFeats = x.shape
        numLabel = len(np.unique(y))

        if depth >= self.maxDepth or numLabel == 1 or numSamples < self.minSampleSplit:
            leafVal = self._commonLabel_(y)
            return Node(value=leafVal)

        featIdx = np.random.choice(numFeats, self.nFeature, replace=False)
        bestFeat = self._bestSplit_(x, y, featIdx)

        children = {}
        uniqueVals = np.unique(x[:, bestFeat])

        for val in uniqueVals:
            idx = np.argwhere(x[:, bestFeat] == val).flatten()
            child = self._grow_(x[idx, :], y[idx], depth+1)
            children[val] = child

        return Node(feature=bestFeat, children=children)

    def _bestSplit_(self, x, y, featIdx):
        # helper function to find the best feature to split the tree

        bestFeat = featIdx[0]
        bestGain = -1

        for feat in featIdx:
            gains = []
            uniqueVals = np.unique(x[:, feat])
            for val in uniqueVals:
                gain = self._infGain_(y, x[:, feat], val)
                gains.append(gain)

            maxGain = max(gains)
            if maxGain > bestGain:
                bestGain = maxGain
                bestFeat = feat

        return bestFeat

    def _commonLabel_(self, y):
        # helper function to find the most common label in a list of labels

        counts = {}
        for label in y:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

        return max(counts, key=counts.get)

    def _infGain_(self, y, xCol, val):
        # helper function to calculate the information gain

        parentEnt = self._entropy_(y)
        leftIdx = np.argwhere(xCol == val).flatten()
        rightIdx = np.argwhere(xCol != val).flatten()
        # print(f'leftIdx: {leftIdx}, rightIdx: {rightIdx}')

        if len(leftIdx) == 0 or len(rightIdx) == 0:

            return 0

        n = len(y)
        nL, nR = len(leftIdx), len(rightIdx)
        entL, entR = self._entropy_(y[leftIdx]), self._entropy_(y[rightIdx])
        childEnt = (nL/n) * entL + (nR/n) * entR

        return parentEnt - childEnt

    def _entropy_(self, y):
        # helper function to calculate the entropy
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy
