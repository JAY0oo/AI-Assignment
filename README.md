
# AI Assignment

This repository contains the implementation of a Decision Tree and a Basic Neural Network Classifier from scratch. The code demonstrates the fundamental principles of machine learning and artificial intelligence.

## Introduction
This project implements two types of classifiers:
1. **Decision Tree Classifier**: A non-parametric supervised learning method used for classification and regression.
2. **Neural Network Classifier**: A basic neural network model built from scratch to perform classification tasks.

## Files
### dt/
- `main.py` : Main driver file for the decision tree.
- `DecsionTree.py`: Contains the implementation of the Decision Tree Classifier.
- `Node.py`: Defines the Node class used in the Decision Tree.

### nnc/
- `main.py`: The main script to run and test the classifiers.
- `nn.py`: Contains the implementation of the Neural Network Classifier.



## Decision Tree Classifier
The Decision Tree Classifier is implemented in `DecsionTree.py`. It includes the following components:
- **Node Class**: Represents each node in the decision tree.
- **Decision Tree Class**: Implements the tree construction and prediction methods.

### Key Features
- Handles both categorical and numerical data.
- Implements common algorithms like ID3, C4.5, or CART for tree construction.
- Pruning methods to prevent overfitting.

## Neural Network Classifier
The Neural Network Classifier is implemented in `nn.py`. It includes the following components:
- **Neural Network Class**: Builds the neural network with layers, activation functions, and backpropagation algorithm.
- **Training and Prediction Methods**: Methods to train the network on data and make predictions.

### Key Features
- Supports multiple layers and activation functions.
- Implements backpropagation for training.
- Designed to handle basic classification tasks.
