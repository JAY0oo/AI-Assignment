import numpy as np
import functions
import math
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)


def printM(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


class Layer:
    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons) * 0.10
        self.biases = self._makeBias_(1, nNeurons)
        self.output = None

    def forwardPass(self, inputs):
        self.output = self._dotProduct_(
            inputs, self.weights, bias=self.biases)

    def _makeBias_(self, rows, cols):
        # default bias = 0
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def _dotProduct_(self, matrix1, matrix2, *, bias=[0]):
        result = [[0 for _ in range(len(matrix2[0]))]
                  for _ in range(len(matrix1))]

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]

        if isinstance(bias[0], list):
            result = [[result[i][j] + bias[0][j]
                      for j in range(len(result[0]))] for i in range(len(result))]
        else:
            result = [[val + bias for val in row] for row in result]

        result = [[round(val, 3) for val in row] for row in result]

        return result

    def printMatrix(self):
        # for debug
        if self.output is not None:
            for row in self.output:
                print(" ".join(map(str, row)))


class Sigmoid:
    """
    Sigmoid activation function.

    sigmoid(x) =  1 / (1 + e^-x)

    Maps any real-valued number to a value between 0 and 1.

    Input: input layer [array]
    Output: sigmoid values of the input layer [array]
    """

    def forward(self, inputs):
        if all(isinstance(i, list) for i in inputs):
            self.output = np.array([[1 / (1 + math.exp(-x))
                                    for x in row] for row in inputs])
        else:
            self.output = np.array([1 / (1 + math.exp(-x)) for x in inputs])


class SoftMax:
    """Applies softmax(x)_i = exp(x_i) / Σ_j exp(x_j) -> converts real numbers into probability distribution used for loss

    Input: output of a Layer [list, nparray]
    Output: softmax values of the output of previous layer [list, nparray (same shape)]
    """

    def forward(self, inputs):
        maxV = max(max(row) for row in inputs)

        exponentialV = [[math.exp(x - maxV)
                        for x in row] for row in inputs]

        sums = [sum(row) for row in exponentialV]

        self.output = np.array([[value / total for value in row]
                               for row, total in zip(exponentialV, sums)])


class Loss:
    def getLoss(self, output, y):
        sampleLoss = self.forward(output, y)
        dfLoss = sum(sampleLoss) / len(sampleLoss)

        return dfLoss


class CatEntLoss(Loss):
    def forward(self, yPred, yTrain):
        samples = len(yTrain)
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)

        if yTrain.ndim == 1:
            return [-math.log(yPredClip[i][yTrain[i]]) for i in range(samples)]
        else:
            return [-math.log(np.sum(yPredClip[i] * yTrain[i])) for i in range(samples)]

    def getAccuracy(self, y_pred, y_true):
        yPredLabels = np.argmax(y_pred, axis=1)

        correctPred = np.sum(yPredLabels == y_true)

        totalPred = len(y_true)

        accuracy = correctPred / totalPred

        return str(round(accuracy, 2) * 100) + "%"


# Load the data
data = pd.read_csv('fashion-mnist_train.csv.gz', compression='gzip')

# Get the pixel values and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Normalize the pixel values
X = X / 255.0

# Convert labels to one-hot encoding
y_encoded = np.eye(10)[y]

# Your existing code
lay1 = Layer(784, 3)
sig1 = Sigmoid()
lay2 = Layer(3, 10)
sig2 = SoftMax()

# Forward pass
lay1.forwardPass(X)
sig1.forward(lay1.output)
lay2.forwardPass(sig1.output)
sig2.forward(lay2.output)

# Calculate loss and accuracy
loss = CatEntLoss()
loss_value = loss.getLoss(sig2.output, y_encoded)
accuracy = loss.getAccuracy(sig2.output, y)

print("Loss: ", loss_value)
print("Accuracy: ", accuracy)
