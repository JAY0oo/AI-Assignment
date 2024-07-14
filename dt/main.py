import pandas as pd
import numpy as np
from Node import Node
from DecsionTree import Tree
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


df = pd.read_csv("car.csv")
#print(df.describe)


def oneHotEncode(df, column):
    # Function to one hot encode a column as the dataset is has many categorical columns

    uniqueVals = df[column].unique()
    for val in uniqueVals:
        df[f"{column}_{val}"] = (df[column] == val).astype(int)

    df.drop(column, axis=1, inplace=True)


def trainTestSplit(X, y, test_size=0.2, random_state=None):
    # Function to randomly split the data into training and testing sets

    if random_state:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - test_size))
    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]

    return X_train, X_test, y_train, y_test


def confusionMatrix(y_true, y_pred):
    # Function to make and calculate a confusion matrix

    classes = sorted(set(y_true))
    matrix = [[0 for _ in classes] for _ in classes]

    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    return matrix


def calPrecisionRecallFs(confusionMatrix):
    # Function to calculate precision, recall and f1 score for the output

    precisions = []
    recalls = []
    f1s = []

    for i in range(len(confusionMatrix)):
        tp = confusionMatrix[i][i]
        fp = sum(confusionMatrix[j][i]
                 for j in range(len(confusionMatrix))) - tp

        fn = sum(confusionMatrix[i]) - tp

        tn = sum(sum(row) for row in confusionMatrix) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) != 0 else 0

        precisions.append(round(precision, 3))
        recalls.append(round(recall, 3))
        f1s.append(round(f1, 3))

    return precisions, recalls, f1s


def printAll(acuracy, matrix, precision, recall, fScore, fitTime, predTime):
    # Prints all required information

    print("INFORMATION: ")
    print(
        f'Training set size: {X_train.shape[0]}\nTesting set size: {X_test.shape[0]}')

    print(
        f"Total Accuracy: {round(acuracy, 2)} = {round(acuracy, 2) * 100}%\n")

    print(f'Precision: {precision}\nF1-Score: {fScore}\nRecall: {recall}\n')

    print("Confusion Matrix:")
    for row in matrix:
        print(row)

    print(f'\nFitting Time: {fitTime}s\nPredicting Time: {predTime}s')


# Data preprocessing
categorical_columns = ['buying', 'maint',
                       'doors', 'persons', 'lug_boot', 'safety']

# One hot encoding
for column in categorical_columns:
    oneHotEncode(df, column)

classMaps = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

df['class'] = df['class'].map(classMaps)

# Features and target
X = df.drop('class', axis=1).values
y = df['class'].values

# Splitting the data
X_train, X_test, y_train, y_test = trainTestSplit(
    X, y, test_size=0.2, random_state=42)

# print(df[:10])

# Training
fitTimeS = time.time()

tree = Tree(maxDepth=10)
tree.fit(X_train, y_train)

fitTimeE = time.time()
fitTime = round(fitTimeE - fitTimeS, 5)

# testing/predicting
predTimeS = time.time()

y_pred = tree.predict(X_test)

predTimeE = time.time()
predTime = round(predTimeE - predTimeS, 5)

# outputs
acuracy = np.sum(y_pred == y_test) / len(y_test)
confMatrix = confusionMatrix(y_test, y_pred)
precision, recall, fScore = calPrecisionRecallFs(confMatrix)

printAll(acuracy, confMatrix, precision, recall, fScore, fitTime, predTime)

# Plotting

trainingSizes = []
trainingAccuracies = []

fractions = np.linspace(0.1, 1.0, 10)

for fraction in fractions:

    trainSize = int(len(X) * fraction)

    X_trainSubset = X_train[:trainSize]
    y_trainSubset = y_train[:trainSize]

    tree.fit(X_trainSubset, y_trainSubset)

    y_pred = tree.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test) * 100

    trainingSizes.append(trainSize)
    trainingAccuracies.append(accuracy)

plt.title('Learning curve')
plt.plot(trainingSizes, trainingAccuracies, marker='o', color="black")
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.grid(True)
plt.show()
