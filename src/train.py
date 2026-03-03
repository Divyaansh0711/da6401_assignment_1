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

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)