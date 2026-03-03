import numpy as np


class Sigmoid:
    def __init__(self):
        self.output_cache = None

    def forward(self, Z):
        """
        Z shape: (batch_size, features)
        """
        A = 1 / (1 + np.exp(-Z))
        self.output_cache = A
        return A

    def backward(self, dA):
        """
        dA: gradient from next layer
        returns: dZ
        """
        A = self.output_cache
        dZ = dA * A * (1 - A)
        return dZ


class Tanh:
    def __init__(self):
        self.output_cache = None

    def forward(self, Z):
        A = np.tanh(Z)
        self.output_cache = A
        return A

    def backward(self, dA):
        A = self.output_cache
        dZ = dA * (1 - A**2)
        return dZ


class ReLU:
    def __init__(self):
        self.input_cache = None

    def forward(self, Z):
        self.input_cache = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        Z = self.input_cache
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    
