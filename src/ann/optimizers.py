import numpy as np


class SGD:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            # L2 regularization
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b
            
class Momentum:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):

            if i not in self.velocity:
                self.velocity[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b)
                }

            vW = self.velocity[i]["W"]
            vb = self.velocity[i]["b"]

            vW = self.beta * vW + layer.grad_W
            vb = self.beta * vb + layer.grad_b

            layer.W -= self.lr * (vW + self.weight_decay * layer.W)
            layer.b -= self.lr * vb

            self.velocity[i]["W"] = vW
            self.velocity[i]["b"] = vb
            
            
class NAG:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):

            if i not in self.velocity:
                self.velocity[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b)
                }

            v_prev_W = self.velocity[i]["W"]
            v_prev_b = self.velocity[i]["b"]

            vW = self.beta * v_prev_W + layer.grad_W
            vb = self.beta * v_prev_b + layer.grad_b

            layer.W -= self.lr * (vW + self.weight_decay * layer.W)
            layer.b -= self.lr * vb

            self.velocity[i]["W"] = vW
            self.velocity[i]["b"] = vb
            
class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.cache = {}

    def step(self, layers):
        for i, layer in enumerate(layers):

            if i not in self.cache:
                self.cache[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b)
                }

            sW = self.cache[i]["W"]
            sb = self.cache[i]["b"]

            sW = self.beta * sW + (1 - self.beta) * (layer.grad_W ** 2)
            sb = self.beta * sb + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * (layer.grad_W / (np.sqrt(sW) + self.eps) + self.weight_decay * layer.W)
            layer.b -= self.lr * (layer.grad_b / (np.sqrt(sb) + self.eps))

            self.cache[i]["W"] = sW
            self.cache[i]["b"] = sb
            
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1

        for i, layer in enumerate(layers):

            if i not in self.m:
                self.m[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                self.v[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

            mW = self.m[i]["W"]
            vW = self.v[i]["W"]
            mb = self.m[i]["b"]
            vb = self.v[i]["b"]

            # First moment
            mW = self.beta1 * mW + (1 - self.beta1) * layer.grad_W
            mb = self.beta1 * mb + (1 - self.beta1) * layer.grad_b

            # Second moment
            vW = self.beta2 * vW + (1 - self.beta2) * (layer.grad_W ** 2)
            vb = self.beta2 * vb + (1 - self.beta2) * (layer.grad_b ** 2)

            # Bias correction
            mW_hat = mW / (1 - self.beta1 ** self.t)
            vW_hat = vW / (1 - self.beta2 ** self.t)

            mb_hat = mb / (1 - self.beta1 ** self.t)
            vb_hat = vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps) + self.weight_decay * layer.W)
            layer.b -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))

            self.m[i]["W"] = mW
            self.m[i]["b"] = mb
            self.v[i]["W"] = vW
            self.v[i]["b"] = vb