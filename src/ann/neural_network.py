import numpy as np
from src.ann.neural_layer import LinearLayer
from src.ann.activations import Sigmoid, Tanh, ReLU


class NeuralNetwork:
    def __init__(self, args):
        """
        args contains:
        - input size (we’ll infer from dataset later)
        - hidden layers
        - activation
        - weight init
        """

        self.layers = []
        self.activations = []

        input_size = 784  # MNIST/Fashion-MNIST flattened
        hidden_sizes = args.hidden_size
        num_layers = args.num_layers
        activation_name = args.activation
        weight_init = args.weight_init

        # Build hidden layers
        prev_size = input_size

        for i in range(num_layers):
            layer = LinearLayer(prev_size, hidden_sizes[i], weight_init)
            self.layers.append(layer)

            if activation_name == "sigmoid":
                self.activations.append(Sigmoid())
            elif activation_name == "tanh":
                self.activations.append(Tanh())
            elif activation_name == "relu":
                self.activations.append(ReLU())
            else:
                raise ValueError("Invalid activation")

            prev_size = hidden_sizes[i]

        # Final output layer (10 classes)
        self.output_layer = LinearLayer(prev_size, 10, weight_init)

    def forward(self, X):
        """
        Forward pass through network.
        Must return logits (NOT softmax).
        """
        for layer, activation in zip(self.layers, self.activations):
            X = layer.forward(X)
            X = activation.forward(X)

        logits = self.output_layer.forward(X)
        return logits

    def backward(self, dLoss):
        """
        Backprop from output to input.
        Must return gradients flowing backward.
        """
        # Output layer backward
        dX = self.output_layer.backward(dLoss)

        # Hidden layers (reverse order)
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            dX = activation.backward(dX)
            dX = layer.backward(dX)

        return dX

    def get_weights(self):
        """
        Required for saving model.
        """
        weights = {
            "hidden": [(layer.W, layer.b) for layer in self.layers],
            "output": (self.output_layer.W, self.output_layer.b)
        }
        return weights

    def set_weights(self, weights):
        """
        Required for loading model.
        """
        for layer, (W, b) in zip(self.layers, weights["hidden"]):
            layer.W = W
            layer.b = b

        self.output_layer.W, self.output_layer.b = weights["output"]