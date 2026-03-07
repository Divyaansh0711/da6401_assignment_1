import copy
import argparse
import numpy as np
import wandb
import json
import os
import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
# FIXED: Added missing metric imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-l", "--loss", type=str, required=True, choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, required=True, choices=["sgd", "momentum", "nag", "rmsprop","adam"])
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True)
    parser.add_argument("-a", "--activation", type=str, required=True, choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init", type=str, required=True, choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    return parser.parse_args()

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
    
import numpy as np
import wandb

def log_sample_images_to_wandb(X_train, y_train, dataset_name):
    print("Logging sample images to W&B...")
    
    # Define class names based on the dataset
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    else: # fashion_mnist
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    columns = ["Class ID", "Class Name", "Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]
    table = wandb.Table(columns=columns)
    
    for class_id in range(10):
        # Find indices for the current class and grab the first 5
        indices = np.where(y_train == class_id)[0]
        sample_indices = indices[:5]
        
        row = [class_id, class_names[class_id]]
        
        for idx in sample_indices:
            # Reshape the flat 784 array back to 28x28 for visualization
            img_array = X_train[idx].reshape(28, 28)
            row.append(wandb.Image(img_array))
            
        table.add_data(*row)
        
    wandb.log({"Dataset Samples": table})
    print("Sample images logged successfully!")
    
def train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, args):
    best_f1 = 0
    best_weights = None
    batch_size = args.batch_size
    
    global_iteration = 0

    for epoch in range(args.epochs):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # FIXED: Accumulate loss to get the epoch average
        running_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward
            logits = model.forward(X_batch)
            
            # Compute dead neuron ratio for first hidden layer
            # Dead neuron monitoring (first hidden layer)
            # activation_layer = model.activations[0]

            # if hasattr(activation_layer, "input_cache"):  
            #     # ReLU case
            #     activations = np.maximum(0, activation_layer.input_cache)
            # else:
            #     # Tanh / Sigmoid case
            #     activations = activation_layer.output_cache
            

            # Only log occasionally to avoid slowing training
            # if epoch % 2 == 0:
            #     fig = plt.figure(figsize=(8,4))
            #     plt.imshow(activations.T, aspect="auto", cmap="viridis")
            #     plt.colorbar()
            #     plt.xlabel("Batch Samples")
            #     plt.ylabel("Neurons")
            #     plt.title("Hidden Layer Activation Heatmap")

            #     wandb.log({"activation_heatmap": wandb.Image(fig)})
            #     plt.close(fig)
            # dead_ratio = np.mean(activations == 0)

            # wandb.log({
            #     "dead_neuron_ratio_layer1": dead_ratio
            # })

            # Loss
            loss = loss_fn.forward(logits, y_batch)
            running_loss += loss
            num_batches += 1
            
            
            
            # Backward
            dLoss = loss_fn.backward()
            model.backward(dLoss)
            
            # # Gradient norm of first hidden layer
            # grad_norm = np.linalg.norm(model.layers[0].grad_W)

            # wandb.log({
            #     "grad_norm_layer1": grad_norm
            # })
            
            

            # Update
            all_layers = model.layers + [model.output_layer]
            optimizer.step(all_layers)

        # Calculate average loss for the epoch
        avg_train_loss = running_loss / num_batches

        # ---- Validation ----
        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_precision = precision_score(y_val, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(y_val, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_preds)

        wandb.log({
            "epoch": epoch + 1,
            "val_accuracy": val_acc,
            "train_loss": avg_train_loss, # Log average instead of last batch
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        print(
            f"Epoch {epoch+1}/{args.epochs}, "
            f"Loss: {avg_train_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Assuming get_weights() returns a copy or a serializable dict. 
            # If it passes by reference, you might need to use copy.deepcopy()
            best_weights = copy.deepcopy(model.get_weights())

    # FIXED: Return the best metrics so the main script can use them
    return model, best_weights, best_f1

if __name__ == "__main__":
    args = get_args()

    wandb.init(project=args.wandb_project, config=vars(args))

    # Load data
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    
    log_sample_images_to_wandb(X_train, y_train, args.dataset)

    # Split validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Create model
    model = NeuralNetwork(args)
    
    # #remove this block
    # # TEMPORARY HACK FOR 2.9 ZEROS RUN - DELETE AFTER
    # # Force all weights and biases to zero
    # for layer in model.layers:
    #     layer.W = np.zeros_like(layer.W)
    #     layer.b = np.zeros_like(layer.b)
    # model.output_layer.W = np.zeros_like(model.output_layer.W)
    # model.output_layer.b = np.zeros_like(model.output_layer.b)

    # Loss & optimizer
    loss_fn = get_loss(args)
    optimizer = get_optimizer(args)

    # Train
    # FIXED: Unpack the new return values from the train function
    model, best_weights, best_f1 = train(
        model, optimizer, loss_fn,
        X_train, y_train,
        X_val, y_val,
        args
    )
        
    
    
    
    # ---- Check against Global Scoreboard ----
    global_scoreboard_path = "src/global_best.json"
    global_best_f1 = 0.0

    # 1. Read the all-time high score (if the file exists)
    if os.path.exists(global_scoreboard_path):
        with open(global_scoreboard_path, "r") as f:
            try:
                global_data = json.load(f)
                global_best_f1 = global_data.get("global_best_f1", 0.0)
            except json.JSONDecodeError:
                pass # If file is empty or corrupted, default to 0.0

    # 2. Compare local best to global best
    if best_weights is not None:
        if best_f1 > global_best_f1:
            print(f"\n🎉 NEW GLOBAL CHAMPION! Local F1 ({best_f1:.4f}) beat the Global F1 ({global_best_f1:.4f})")
            
            # Save the new winning weights
            np.save(args.model_path, best_weights)
            
            # Save the new winning configuration
            best_config = {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "weight_init": args.weight_init,
                "best_val_f1": best_f1
            }
            config_path = args.model_path.replace(".npy", "_config.json")
            with open(config_path, "w") as f:
                json.dump(best_config, f, indent=4)
            
            # Update the global scoreboard for the next player
            with open(global_scoreboard_path, "w") as f:
                json.dump({"global_best_f1": best_f1}, f, indent=4)
                
            print(f"Files successfully updated at {args.model_path} and {config_path}")
            
        else:
            print(f"\n❌ Local best ({best_f1:.4f}) did not beat the Global best ({global_best_f1:.4f}). Discarding model.")
    else:
        print("\nWarning: No best model found in this run.")

    wandb.finish()