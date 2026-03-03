import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["mean_squared_error", "cross_entropy"])

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop","adam"])

    parser.add_argument("-lr", "--learning_rate", type=float, required=True)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-nhl", "--num_layers", type=int, required=True)

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True)

    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-w_i", "--weight_init", type=str, required=True,
                        choices=["random", "xavier"])

    parser.add_argument("-w_p", "--wandb_project", type=str, required=True)

    return parser.parse_args()


import numpy as np
import wandb

from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam

def load_data(dataset_name):
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    return X_train, y_train, X_test, y_test

def get_optimizer(args):
    if args.optimizer == "sgd":
        return SGD(args.learning_rate, args.weight_decay)
    elif args.optimizer == "momentum":
        return Momentum(args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        return NAG(args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        return RMSProp(args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        return Adam(args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid optimizer")
    
def get_loss(args):
    if args.loss == "cross_entropy":
        return CrossEntropyLoss()
    elif args.loss == "mean_squared_error":
        return MeanSquaredError()
    else:
        raise ValueError("Invalid loss")
    
def train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, args):

    best_f1 = 0
    best_weights = None

    batch_size = args.batch_size

    for epoch in range(args.epochs):

        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, len(X_train), batch_size):

            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward
            logits = model.forward(X_batch)

            # Loss
            loss = loss_fn.forward(logits, y_batch)

            # Backward
            dLoss = loss_fn.backward()
            model.backward(dLoss)

            # Update
            all_layers = model.layers + [model.output_layer]
            optimizer.step(all_layers)

        # ---- Validation ----
        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)

        val_acc = accuracy_score(y_val, val_preds)

        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_acc,
            "train_loss": loss
        })

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

args = get_args()

wandb.init(project=args.wandb_project, config=vars(args))

# Load data
X_train, y_train, X_test, y_test = load_data(args.dataset)

# Split validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# Create model
model = NeuralNetwork(args)

# Loss & optimizer
loss_fn = get_loss(args)
optimizer = get_optimizer(args)

# Train
model = train(model, optimizer, loss_fn,
              X_train, y_train,
              X_val, y_val,
              args)

# Save model
best_weights = model.get_weights()
np.save("src/best_model.npy", best_weights)

wandb.finish()