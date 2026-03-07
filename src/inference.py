import argparse
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["mean_squared_error", "cross_entropy"])

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-lr", "--learning_rate", type=float, required=True)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-nhl", "--num_layers", type=int, required=True)

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True)

    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-w_i", "--weight_init", type=str, required=True,
                        choices=["random", "xavier"])

    parser.add_argument("-w_p", "--wandb_project", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")

    return parser.parse_args()

# can use below for sanity check
# if __name__ == "__main__":
#     args = get_args()
#     print(args)

import argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def load_data(dataset):
    if dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test.reshape(-1, 784) / 255.0
    return X_test, y_test


def main(args):

    # Load dataset
    X_test, y_test = load_data(args.dataset)

    # Build model
    model = NeuralNetwork(args)

    # Load weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Forward pass
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    # Metrics
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")
    
    # wandb.init(project=args.wandb_project, name="Final_Test_Evaluation")
    # # Initialize WandB run if you haven't already in this script

    # # ==========================================
    # # 2.8.1: Confusion Matrix
    # # ==========================================
    # cm = confusion_matrix(y_test, preds)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    # disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    # ax_cm.set_title(f"Confusion Matrix - {args.dataset}")
    
    # # Log to W&B
    # wandb.log({"Confusion Matrix": wandb.Image(fig_cm)})
    # plt.close(fig_cm)

    # # ==========================================
    # # 2.8.2: Creative Error Analysis (Failures)
    # # ==========================================
    # # Find indices where predictions don't match ground truth
    # incorrect_indices = np.where(preds != y_test)[0]
    
    # # Setup a 2x5 grid for the first 10 mistakes
    # fig_err, axes = plt.subplots(2, 5, figsize=(15, 6))
    # fig_err.suptitle("Model Misclassifications", fontsize=16)
    
    # for i, ax in enumerate(axes.flat):
    #     if i < len(incorrect_indices):
    #         idx = incorrect_indices[i]
    #         # Reshape the flattened 784 array back to 28x28 image
    #         img = X_test[idx].reshape(28, 28)
            
    #         ax.imshow(img, cmap='gray')
    #         ax.set_title(f"True: {y_test[idx]} | Pred: {preds[idx]}")
    #         ax.axis('off')
            
    # plt.tight_layout()
    
    # # Log the error analysis plot to W&B
    # wandb.log({"Error Analysis": wandb.Image(fig_err)})
    # plt.close(fig_err)
    
    # wandb.finish()

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


if __name__ == "__main__":

    args = get_args()
    main(args)