from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
     
from utils import load_data, compute_accuracy

class Random_Forest(object):

    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, Y):
        """ 
        Fit the Random Forest model on labeled training data.
        
        Args:
            X: Pandas dataframe of shape [num_examples, num_features]
            Y: Pandas dataframe of shape [num_examples, 1]
        """
        # Create train/val splits
        train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.6)
        
        # Fit model
        self.model.fit(train_x, train_y)

        # Check accuracy of model
        predictions = self.model.predict(val_x)
        print("Model predicted network traffic with an accuracy of", accuracy_score(predictions, val_y) * 100, '%')

    def predict(self, X):
        """ 
        Predict network traffic type
        
        Args: 
            X: Pandas dataframe of shape [num_examples, num_features]
        Returns:
            A dense array of ints with shape [num_examples, 1]
        """
        return self.model.predict(X)
    

class Naive_Bayes(object):
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, Y):
        """ 
        Fit the Random Forest model on labeled training data.
        
        Args:
            X: Pandas dataframe of shape [num_examples, num_features]
            Y: Pandas dataframe of shape [num_examples, 1]
        """
        # Create train/val splits
        train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.5)
        
        # Fit model
        self.model.fit(train_x, train_y)

        # Check accuracy of model
        predictions = self.model.predict(val_x)
        print("Model predicted network traffic with an accuracy of", accuracy_score(predictions, val_y) * 100, '%')

    def predict(self, X):
        """ 
        Predict network traffic type
        
        Args: 
            X: Pandas dataframe of shape [num_examples, num_features]
        Returns:
            A dense array of ints with shape [num_examples, 1]
        """
        return self.model.predict(X)

class DeepInsightCNN(object):
    # TODO: implement
    def __init__(self):
        pass

    def train(self, X, Y):
        pass

    def predict(self, X):
        pass
