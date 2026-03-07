import numpy as np
import argparse

from ann.neural_network import NeuralNetwork

best_config= argparse.Namespace(
            {
    "dataset": "mnist",
    "epochs": 10,
    "batch_size": 64,
    "loss": "mean_squared_error",
    "optimizer": "rmsprop",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "num_layers": 2,
    "hidden_size": [
        128,
        64
    ],
    "activation": "relu",
    "weight_init": "xavier",
    "best_val_f1": 0.9807586882923325
}
        )

model = NeuralNetwork(best_config)

weights = np.load("best_model.npy", allow_pickle=True).item()

model.set_weights(weights)

