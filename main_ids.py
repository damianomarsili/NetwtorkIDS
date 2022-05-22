import os 
import argparse as ap
import pickle
import numpy as np

# Import model
from utils import load_data, compute_accuracy

def get_args():
    """Parse arguments for IDS model parameters. """ 
    # TODO: Update description based on implementation.
    p = ap.ArgumentParser(description = "A ___ based Intrusion detection system for network traffic diagnostics.") 

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], help=
            "Set mode: train or test.")
    p.add_argument("--model", type=str, required=True, choices=["k-means"], help="Select which model to use.")
    p.add_argument("--num_clusters", type=int, help="Number of clusters (only for k-means)")
    p.add_argument("--train-data", type=str, help="Path to training data")
    p.add_argument("--test-data", type=str, help="Path to test data")
    p.add_argument("--model-file", type=str, required=True, 
        help="Path where model will be saved or loaded")
    
    # Model Hyperparameters - TODO: figure out model hyperparameters

    p.add_argument("--training-iterations", type=int, help="Number of training iterations")

    return p.parse_args()


def check_args(args): # TODO - modify params based on hyperparameters identified.
        mandatory_args = {'mode', 'model', 'model_file', 'test_data', 'train_data', 'training_iterations'}
        if not mandatory_args.issubset(set(dir(args))):
                raise Exception("Incomplete set of arguments.")

        if args.mode.lower() == 'train':
                if args.model_file is None:
                        raise Exception("Must specify model file during iteration")
                if args.train_data is None:
                        raise Exception("Must specify path to training data for training.")
                elif not os.path.exists(args.train_data):
                        raise Exception("Path specified by train-data does not exist")
        elif args.mode.lower() == 'test':
                if not os.path.exists(args.model_file):
                        raise Exception("Path specified by model-file does not exist.")
                if args.test_data is None:
                        raise Exception("Must specify test data for testing")
                elif not os.path.exists(args.test_data):
                        raise Exception("Path speicifed by test-data does not exist")
        else:
                raise Exception("Invalid mode.")
        
        if args.model.lower() == 'k_means':
                if args.num_clusters is None:
                        raise Exception("Must specify number of clusters for K-means")
        else:
                raise Exception("Invalid model.")

def train_k_means(args):
        """Fit a models parameters to labeled network traffic given the parameters specified in args"""

        # Load the training data
        X, Y = load_data(args.train_data)

        # Build the model
        model = None #TODO: define

        # Run the training loop
        model.fit(X = X, Y = Y, k = args.num_clusters, iterations = args.training_iterations)

        # Save the model file
        pickle.dump(model, open(args.model_file, 'wb'))

def test(args):
        """Make predictions over the specified test dataset, and store the predictions."""
        # Load dataset and model
        X, Y = load_data(args.test_data)
        model = pickle.load(open(args.model_file, 'rb'))

        # Predict labels for test dataset
        preds = model.predict(X)

        # Compute model accuracy
        acc = compute_accuracy(Y, preds) 
        print('Model classified network traffic with an accuracy of ', acc, "%")

if __name__ == "__main__":
        args = get_args()
        check_args(args)

        if args.mode.lower() == 'train':
                if args.model.lower() == 'k_means':    
                        train_k_means(args)
        if args.mode.lower() == 'test':
                test(args)
                