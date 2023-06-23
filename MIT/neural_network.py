import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # initialize a matrix n_inputs x n_neurons
        self.biases = np.zeros((1, n_neurons))

    def call(self, inputs):
        z = np.matmul(inputs, self.weights) + self.biases
        output = sigmoid(z)
        return output


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

