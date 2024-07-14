import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math
import time

np.random.seed(0)


class Layer:
    """
    A layer class in a neural network

    Attributes:
        weights : The weights of the layer.
        biases : The biases of the layer.
        output : The output of the layer.
        inputs : The inputs to the layer.
        dweights : The gradients of the weights.
        dbiases : The gradients of the biases.
        dinputs : The gradients of the inpus.

    Methods:
        __init__: Initializes the layer with random weights and zero biases.
        forwardPass: Forward Propagation.
        backwardPass: Backward propegation, updates weights and biases.
    """

    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons) * 0.10
        self.biases = np.zeros((1, nNeurons))
        self.output = None

    def forwardPass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backwardPass(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        self.dinputs = np.dot(dvalues, self.weights.T)


class Sigmoid:
    """
    Sigmoid Activation Function 
    Formula: σ(x) = 1 / (1 + exp(-x))
    """

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))


class ReLU:
    # FOR TESTING ONLY
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0


class SoftMax:
    """
    SoftMax activation function. Takes a vector of inputs and computes the probabilities of each class.

    Formula: softmax(x_i) = exp(x_i) / Σ(exp(x_j))

    Mehods:
    - forward(): Performs the forward pass. 
    - backward(): Performs the backward pass.

    Attributes:
    - dinputs: The gradient of the SoftMax activation function with respect to the inputs.
    """

    def forward(self, inputs):
        expVal = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = expVal / np.sum(expVal, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class CatEntLoss:
    """    
    Categorical Cross-Entropy Loss.

    Implements the forward and backward pass for the categorical cross-entropy loss function.
    Also has a method to calculate the accuracy of the predictions.

    Methods:
        forward(): Calculates the loss given the predicted values and the true labels.
        getAccuracy(): Calculates the accuracy of the predictions.
        backward(): Calculates the gradients of the loss with respect to the inputs.

    """

    def forward(self, yPred, yTrain):
        samples = len(yTrain)
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)

        if len(yTrain.shape) == 1:
            correct_log_probs = -np.log(yPredClip[range(samples), yTrain])
        else:
            correct_log_probs = -np.log(np.sum(yPredClip*yTrain, axis=1))

        loss = np.mean(correct_log_probs)
        return loss

    def getAccuracy(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        #print(f'{accuracy}\t\t{predictions}')
        return accuracy * 100

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


def trainModel(*, inputDim, hiddenDim, outputDim, epochs, learningRate, trainingDf, allowOverfit=True):
    # Trains the model on the given training dataset. Returns the trained model and the accuracy of the model on the training set.
    # Can allow overfitting for reporting purposes.

    trainingDf = pd.read_csv("fashion-mnist_test.csv.gz", compression="gzip")

    x_train = trainingDf.drop("label", axis=1).values
    y_train = trainingDf["label"].values

    # normalizing
    X_train = x_train / 255.0

    # 1hot encode
    y_trainEncoded = np.eye(10)[y_train]

    # Initialize layers and loss function
    hiddenLayer = Layer(inputDim, hiddenDim)
    activationSig = Sigmoid()
    outputLayer = Layer(hiddenDim, outputDim)
    activationSoft = SoftMax()
    loss = CatEntLoss()

    accuracyList = []
    prevAccuracy = 0.0

    # Training loop
    for epoch in range(epochs):
        hiddenLayer.forwardPass(X_train)
        activationSig.forward(hiddenLayer.output)
        outputLayer.forwardPass(activationSig.output)
        activationSoft.forward(outputLayer.output)

        loss_value = loss.forward(activationSoft.output, y_trainEncoded)
        accuracy = loss.getAccuracy(activationSoft.output, y_train)

        print(
            f'Epoch: {epoch}, loss: {loss_value}, accuracy: {round(accuracy, 3)}%')

        loss.backward(activationSoft.output, y_trainEncoded)
        activationSoft.backward(loss.dinputs)
        outputLayer.backwardPass(activationSoft.dinputs, learningRate)
        activationSig.backward(outputLayer.dinputs)
        hiddenLayer.backwardPass(activationSig.dinputs, learningRate)

        accuracyList.append(accuracy)

        if accuracy > prevAccuracy:
            prevAccuracy = accuracy

        # early stop
        if not allowOverfit and accuracy < prevAccuracy:
            break

    print(prevAccuracy)
    return hiddenLayer, activationSig, outputLayer, activationSoft, loss, accuracyList


def testModel(hiddenLayer, activationSig, outputLayer, activationSoft, loss, testingDf, batch_size=32):
    # Test the model on the given testing dataset after training. Returns the loss and accuracy of the model on the testing set.

    testingDf = pd.read_csv(testingDf, compression="gzip")

    x_test = testingDf.drop("label", axis=1).values
    y_test = testingDf["label"].values

    # normalizing
    X_test = x_test / 255.0

    # 1hot encode
    y_testEncoded = np.eye(10)[y_test]

    testLosses = []
    testAccuracies = []
    prevAccuracy = None

    indices = np.arange(X_test.shape[0])
    for start_idx in range(0, X_test.shape[0] - batch_size + 1, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]

        hiddenLayer.forwardPass(X_test[batch_idx])
        activationSig.forward(hiddenLayer.output)
        outputLayer.forwardPass(activationSig.output)
        activationSoft.forward(outputLayer.output)

        testLoss = loss.forward(activationSoft.output,
                                y_testEncoded[batch_idx])
        testAccuracy = loss.getAccuracy(
            activationSoft.output, y_test[batch_idx])

        testLosses.append(testLoss)
        testAccuracies.append(testAccuracy)

    avgLoss = np.mean(testLosses)
    avgAccuracy = np.mean(testAccuracies)
    highestAccuracy = math.ceil(np.max(testAccuracies))
    minLoss = np.min(testLosses)

    print(
        f'Average test loss: {avgLoss}, Average test accuracy: {round(avgAccuracy, 3)}%, Maximum Accuracy: {highestAccuracy}%, Lowest Loss: {minLoss}')
    return testLosses, testAccuracies


# command line arguments
args = sys.argv
#print(args)

# Hyper Parameters
nInputs = int(args[1])
nHidden = int(args[2])
nOutput = int(args[3])
trainingSet = args[4]
testingSet = args[5]
epoch = 50
learnRate = 0.01

print(
    f'Inputs:\nNumber Of Input Neurons: {nInputs}\nNumber Of Hidden Neurons: {nHidden}\nNumber Of Output Neurons: {nOutput}\nEpochs: {epoch}\nTraining Model....\n')

# Training
trainTimeS = time.time()

hiddenLayer, activationSig, outputLayer, activationSoft, loss, accuracy = trainModel(inputDim=nInputs, hiddenDim=nHidden, outputDim=nOutput,
                                                                                     epochs=epoch, learningRate=learnRate, trainingDf=trainingSet, allowOverfit=False)

trainTimeE = time.time()

plt.plot(range(1, len(accuracy) + 1), accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve: (n = {learnRate})')
plt.show()


# Testing
print("Testing....")
testTimeS = time.time()

testLoss, testAccuracy = testModel(
    hiddenLayer, activationSig, outputLayer, activationSoft, loss, testingSet)

testTimeE = time.time()

plt.plot(range(len(testAccuracy)), testAccuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Learning Curve After Training")
plt.show()

print(
    f'Training time: {math.ceil(trainTimeE - trainTimeS)}s, Testing Time: {round(testTimeE - testTimeS, 4)}s')
