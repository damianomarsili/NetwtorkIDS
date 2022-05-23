import os 
import argparse as ap
import pickle
import numpy as np

import models

# Import model
from utils import load_data, compute_accuracy

def get_args():
    """Parse arguments for IDS model parameters. """ 
    p = ap.ArgumentParser(description = "A ___ based Intrusion detection system for network traffic diagnostics.") 

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help=
            "Set mode: train or predict.")
    p.add_argument("--model", type=str, required=True, choices=["random-forest", "naive-bayes"], help="Select which model to use.")
    p.add_argument("--train-data", type=str, help="Path to training data")
    p.add_argument("--predict-data", type=str, help="Path to predict data")
    p.add_argument("--model-file", type=str, required=True, 
        help="Path where model will be saved or loaded")
    
    return p.parse_args()


def check_args(args):
        mandatory_args = {'mode', 'model', 'model_file', 'predict_data', 'train_data'}
        if not mandatory_args.issubset(set(dir(args))):
                raise Exception("Incomplete set of arguments.")

        if args.mode.lower() == 'train':
                if args.model_file is None:
                        raise Exception("Must specify model file during iteration")
                if args.train_data is None:
                        raise Exception("Must specify path to training data for training.")
                elif not os.path.exists(args.train_data):
                        raise Exception("Path specified by train-data does not exist")
        elif args.mode.lower() == 'predict':
                if not os.path.exists(args.model_file):
                        raise Exception("Path specified by model-file does not exist.")
                if args.predict_data is None:
                        raise Exception("Must specify predict data for testing")
                elif not os.path.exists(args.predict_data):
                        raise Exception("Path speicifed by predict-data does not exist")
        else:
                raise Exception("Invalid mode.")
        
        if args.model.lower() not in ['random-forest','naive-bayes']:
             raise Exception("Invalid model.")

def train(args):
        """Fit a models parameters to labeled network traffic given the parameters specified in args"""
        # Load the training data
        X, Y = load_data(args.train_data)

        # Build the model
        if (args.model.lower() == 'random-forest'):
                model = models.Random_Forest()
        elif (args.model.lower() == 'naive-bayes'):
                model = models.Naive_Bayes()

        # Run the training loop
        model.fit(X = X, Y = Y)

        # Save the model file
        pickle.dump(model, open(args.model_file, 'wb'))

def predict(args):
        """Make predictions over the specified predict dataset, and store the predictions."""
        # Load dataset and model
        X, Y = load_data(args.predict_data)
        model = pickle.load(open(args.model_file, 'rb'))

        # Predict labels for predict dataset
        preds = model.predict(X)

        # Store predictions
        np.savetxt('preds.csv', preds, delimiter=',')

if __name__ == "__main__":
        args = get_args()
        check_args(args)

        print(args)

        if args.mode.lower() == 'train':
                train(args)
        if args.mode.lower() == 'predict':
                predict(args)
                