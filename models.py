from sklearn.ensemble import RandomForestClassifier
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
            X: TODO: determine X type
            Y: TODO: determine Y type
        Returns:
            A dense array of ints with shape [num_examples]
        """
        raise NotImplementedError()
    

