import numpy as np


class LinearLayer:
    def __init__(self, in_features, out_features, weight_init="random"):
        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(2 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * limit
        else:
            raise ValueError("Invalid weight initialization method")

        self.b = np.zeros((1, out_features))

        # Required for autograder
        self.grad_W = None
        self.grad_b = None

        # Cache for backward pass
        self.input_cache = None

    def forward(self, X):
        """
        X shape: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        self.input_cache = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        """
        dZ shape: (batch_size, out_features)

        Returns:
        dX shape: (batch_size, in_features)
        """

        X = self.input_cache
        batch_size = X.shape[0]

        # Gradients
        self.grad_W = np.dot(X.T, dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # Gradient w.r.t input
        dX = np.dot(dZ, self.W.T)

        return dX
    
    
    
    
