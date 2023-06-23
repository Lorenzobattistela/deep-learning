
# we need to calculate the next step using partial derivatives and update the weights correctly

# initialize weights randomly
# loop until convergence:
    # compute gradient dj(W)/dW -> backpropagation
    # update weights
# return weights
from .neural_network import Layer
from .losses import MeanSquaredError
import numpy as np

# how does a small change in one weight affects the final loss?
class GradientDescent:
    def __init__(self, layer: Layer, num_iterations: int, y_true, y_pred, inputs, m) -> None:
        # intialize weigths randomly
        self.weights = layer.weights
        self.x = inputs
        self.iterations = num_iterations
        self.y_pred = y_pred
        self.y_true = y_true
        self.m = m # where m is the number of examples
    
    def compute_gradient(self):
        for i in range(self.iterations):
            loss = MeanSquaredError(y_pred=self.y_pred, y_true=self.y_true)
            loss = loss.call()
            print(f"Iteration {i}: Loss: {loss}")
            gradient = np.dot(self.x.transpose(), loss) / self.m
            self.weights = self.weights * gradient
    
